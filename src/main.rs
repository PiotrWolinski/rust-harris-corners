use clap::Parser;
use image::buffer::ConvertBuffer;
use image::io::Reader as ImageReader;
use image::{GrayImage, Rgb, RgbImage};
use imageproc::drawing::draw_filled_circle_mut;
use ndarray::{array, s, Array, Array2, Zip};
use ndarray_conv::*;
use ndarray_ndimage::*;
use ndarray_stats::*;
use rayon::prelude::*;
use show_image;
use std::path;
use std::time::Instant;

const RED: Rgb<u8> = image::Rgb([255, 0, 0]);

#[derive(Parser)]
struct Cli {
    #[arg(short = 'n', long, default_value_t = 10)]
    num_runs: u32,

    #[arg(short = 'f', long, default_value = "data/000000.png")]
    file_path: std::path::PathBuf,

    #[arg(long, default_value_t = 200)]
    max_keypoints: usize,

    #[arg(short = 's', long, default_value_t = 8)]
    surpression_window: usize,

    #[arg(short = 'r', long, default_value_t = 3)]
    keypoint_radius: i32,

    #[arg(short = 'p', long, default_value_t = 9)]
    patch_size: usize,

    #[arg(long, default_value_t = 0.08)]
    kappa: f32,
}

fn read_image_rgb(file_path: &path::PathBuf) -> RgbImage {
    ImageReader::open(file_path)
        .expect("Failed to read the image!")
        .decode()
        .expect("Failed to decode the image!")
        .to_rgb8()
}

fn run_harris(img: GrayImage, surp_size: &usize, patch_size: &usize, max_keypoints: &usize, kappa: &f32) -> Vec<(usize, usize)>{
    let img_width = &img.width();
    let img_height = &img.height();
    let sobel_x = array![[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]].mapv(|x| x as f32);
    let sobel_y = array![[-1, -2, -1], [0, 0, 0], [1, 2, 1]].mapv(|x| x as f32);
    let ones_patch: Array2<f32> = Array2::ones((*patch_size, *patch_size));

    let img_to_arr = Array::from_shape_vec(
        (*img_height as usize, *img_width as usize),
        img.clone().into_vec(),
    )
    .expect("Cannot convert image to ndarray!")
    .mapv(|x: u8| x as f32);

    // Get the derivatives in x and y
    let mut i_x: Array2<f32> = img_to_arr
        .conv(&sobel_x, ConvMode::Valid, PaddingMode::Replicate)
        .expect("Failed to conv");
    let mut i_y: Array2<f32> = img_to_arr
        .conv(&sobel_y, ConvMode::Valid, PaddingMode::Replicate)
        .expect("Failed to conv");

    let i_x_y = &i_x * &i_y;

    i_x.par_mapv_inplace(|x| x.powf(2.0));
    i_y.par_mapv_inplace(|x| x.powf(2.0));

    // Calculate elements of the second moment matrix
    let mut i_x_sum = i_x
        .conv(&ones_patch, ConvMode::Valid, PaddingMode::Replicate)
        .expect("Failed to conv");
    let i_y_sum = i_y
        .conv(&ones_patch, ConvMode::Valid, PaddingMode::Replicate)
        .expect("Failed to conv");
    let i_x_y_sum = i_x_y
        .conv(&ones_patch, ConvMode::Valid, PaddingMode::Replicate)
        .expect("Failed to conv");

    Zip::from(&mut i_x_sum)
        .and(&i_y_sum)
        .and(&i_x_y_sum)
        .par_for_each(|x, &y, &x_y| {
            *x = (*x * y) - x_y.powf(2.0) - (kappa * (*x + y).powf(2.0));
        });

    let pad_size = (patch_size / 2) + 1;

    let mut score = pad(&i_x_sum, &[[pad_size, pad_size]], PadMode::Constant(0.0));

    let mut selected_keypoints_number = 0;
    let mut selected_keypoints: Vec<(usize, usize)>  = vec![(usize::MAX, usize::MAX); *max_keypoints];

    // Perform non-maxima surpression around the highest responses
    // and mark keypoints on the image
    while &selected_keypoints_number < max_keypoints {
        let max_id: (usize, usize) = score.argmax().unwrap();

        score
            .slice_mut(s![
                max_id.0 - surp_size..max_id.0 + surp_size,
                max_id.1 - surp_size..max_id.1 + surp_size
            ])
            .fill(0.0);
        
        // Save keypoints to the array to return them later
        selected_keypoints.push(max_id);
        selected_keypoints_number += 1;
    }

    return selected_keypoints;
}

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();

    println!("Provided path: {:?}", args.file_path);

    // let cwd: path::PathBuf = env::current_dir().expect("Failed to get CWD!");
    let data_path: path::PathBuf = args.file_path;

    println!("Chosen image: {}", data_path.display());
    
    let original_rgb_image = read_image_rgb(&data_path);
    let original_image: GrayImage = original_rgb_image.clone().convert();
    
    let t0 = Instant::now();
    let selected_keypoints = run_harris(original_image, &args.surpression_window, &args.patch_size, &args.max_keypoints, &args.kappa);
    let duration = t0.elapsed();
    
    let mut img_to_show = original_rgb_image;

    for keypoint in selected_keypoints {
        draw_filled_circle_mut(
            &mut img_to_show,
            (keypoint.1 as i32, keypoint.0 as i32),
            args.keypoint_radius,
            RED,
        );
    }

    println!("Harris corners computation took {:?}", duration);

    let window =
        show_image::create_window("image", Default::default()).expect("Cannot create window!");
    window
        .set_image("image-window", img_to_show)
        .expect("Cannot set image!");

    // Print keyboard events until Escape is pressed, then exit.
    for event in window.event_channel()? {
        if let show_image::event::WindowEvent::KeyboardInput(event) = event {
            if event.input.key_code == Some(show_image::event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                break;
            }
        }
    }

    Ok(())
}
