use chrono::prelude::*;
use clap::Parser;
use image::ImageReader;
use rand::Rng;
use rumpus::estimator::pattern_match::Searcher;
use rumpus::prelude::*;
use serde::{Deserialize, Serialize};
use sguaba::{engineering::Orientation, systems::Wgs84};
use std::{io::Read, path::PathBuf};
use uom::{
    si::{
        angle::degree,
        f64::{Angle, Length},
    },
    ConstZero,
};

#[derive(Parser)]
struct Cli {
    image: PathBuf,
    params: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct SimulationParams {
    pixel_size: Length,
    focal_length: Length,
    latitude: Angle,
    longitude: Angle,
    time: DateTime<Utc>,
    max_iters: usize,
}

fn main() {
    let args = Cli::parse();
    let params = parse_params(&args.params).expect("readable and parsable params");

    let raw_image = ImageReader::open(&args.image)
        .expect("a valid image path")
        .decode()
        .expect("a supported image format")
        .into_luma8();

    let (width, height) = raw_image.dimensions();
    let estimate = IntensityImage::from_bytes(width as u16, height as u16, &raw_image.into_raw())
        .expect("image dimensions are even")
        .rays(params.pixel_size, params.pixel_size)
        .estimate(PatternMatch::new(
            Lens::from_focal_length(params.focal_length)
                .expect("focal length is greater than zero"),
            SkyModel::from_wgs84_and_time(
                Wgs84::builder()
                    .latitude(params.latitude)
                    .expect("latitude is between -90 and 90")
                    .longitude(params.longitude)
                    .altitude(Length::ZERO)
                    .build(),
                params.time,
            ),
            RandomSearch::new(rand::rng()),
            params.max_iters,
        ));

    println!("{:#?}", estimate.to_tait_bryan_angles());
}

fn parse_params(path: &PathBuf) -> Option<SimulationParams> {
    let mut buffer = String::new();
    std::fs::File::open(path)
        .ok()?
        .read_to_string(&mut buffer)
        .ok()?;
    toml::from_str(&buffer).ok()
}

struct RandomSearch<R> {
    rng: R,
}

impl<R> RandomSearch<R> {
    fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: Rng> Searcher for RandomSearch<R> {
    type Iter = RandomSearchIter<R>;

    fn orientations(self) -> Self::Iter {
        RandomSearchIter { rng: self.rng }
    }
}

struct RandomSearchIter<R> {
    rng: R,
}

impl<R: Rng> Iterator for RandomSearchIter<R> {
    type Item = Orientation<CameraEnu>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(
            Orientation::<CameraEnu>::tait_bryan_builder()
                .yaw(Angle::new::<degree>(self.rng.random_range(0.0..360.0)))
                .pitch(Angle::new::<degree>(self.rng.random_range(-5.0..5.0)))
                .roll(Angle::new::<degree>(self.rng.random_range(-5.0..5.0)))
                .build(),
        )
    }
}
