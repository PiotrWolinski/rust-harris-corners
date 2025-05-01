# Harris corner detector in rust

## Intro

This is my first project in Rust and even though I know how Harris corner detector works and how it should be implemented, it may not be the most appropriate way to do it *in Rust*. Feel free to point out anything that can make this tiny project better.

Code is not by any means in its final shape, but the core functionality (harris score) works just fine.

## Progress

- [x] Perform non-maximum surpression and establish keypoints  
- [x] Mark keypoints in some visible way on the image
- [x] Make keypoints more visible
- [ ] Add CLI to provide arguments to the app
- [ ] Store the results of the detections for further operations on them
- [ ] Add keypoints descriptor for measuring similarity between them
- [ ] Allow video as an input
- [ ] Add option to use at as a CLI app
