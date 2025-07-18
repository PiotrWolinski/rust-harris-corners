use image::buffer::ConvertBuffer;
use image::io::Reader as ImageReader;
use image::{GrayImage, Rgb, RgbImage};
use ndarray::{s, array, Array, Array2, Zip};
use ndarray_conv::*;
use ndarray_ndimage::*;
use ndarray_stats::*;
use show_image;
use std::env;
use std::path;
use std::time::Instant;
use rayon::prelude::*;
use imageproc::drawing::draw_filled_circle_mut;


fn read_image_rgb(file_path: &path::PathBuf) -> RgbImage {
    ImageReader::open(file_path)
        .expect("Failed to read the image!")
        .decode()
        .expect("Failed to decode the image!")
        .to_rgb8()
}


#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    const DATA_DIR: &str = "data";
    const IMG_NAME: &str = "000000.png";
    const MAX_KEYPOINTS: i32 = 200;
    const SURP_SIZE: usize = 8;
    const RED:Rgb<u8> = image::Rgb([255, 0, 0]);
    const KEYPOINT_RADIUS: i32 = 3;

    // let paths = data_path.read_dir().unwrap();
    // for file in paths {
    //     let file_name = file.unwrap().file_name();
    //     println!("{}", file_name.into_string().unwrap());
    // }

    let cwd: path::PathBuf = env::current_dir().expect("Failed to get CWD!");
    let data_path: path::PathBuf = cwd.join(DATA_DIR).join(IMG_NAME);

    println!("Chosen image: {}", data_path.display());

    let kappa: f32 = 0.08;
    let patch_size: usize = 9;

    let original_rgb_image = read_image_rgb(&data_path);
    let original_image: GrayImage = original_rgb_image.clone().convert();

    let img_width = &original_image.width();
    let img_height = &original_image.height();
    let sobel_x = array![[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]].mapv(|x| x as f32);
    let sobel_y = array![[-1, -2, -1], [0, 0, 0], [1, 2, 1]].mapv(|x| x as f32);
    let ones_patch: Array2<f32> = Array2::ones((patch_size, patch_size));

    let img_to_arr =
        Array::from_shape_vec((*img_height as usize, *img_width as usize), original_image.clone().into_vec())
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
    let mut img_to_show = original_rgb_image;

    // Perform non-maxima surpression around the highest responses
    // and mark keypoints on the image
    while &selected_keypoints_number < &MAX_KEYPOINTS {
        let max_id: (usize, usize) = score.argmax().unwrap();
        draw_filled_circle_mut(&mut img_to_show, (max_id.1 as i32, max_id.0 as i32), KEYPOINT_RADIUS, RED);

        score.slice_mut(s![max_id.0-SURP_SIZE..max_id.0+SURP_SIZE, max_id.1-SURP_SIZE..max_id.1+SURP_SIZE]).fill(0.0);

        selected_keypoints_number += 1;
    }

    let duration = start.elapsed();
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
