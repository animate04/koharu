use anyhow::{Result, ensure};
use image::DynamicImage;

use super::{KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask};

pub(super) fn inference(
    image: &DynamicImage,
    mut infer: impl FnMut(&DynamicImage) -> Result<KoharuLayoutDetections>,
) -> Result<KoharuLayoutDetections> {
    let (width, height) = (image.width(), image.height());
    ensure!(width > 0 && height > 0, "cannot segment an empty image");
    // Match the existing comic detector's long-page threshold; crop before the
    // RF-DETR square resize so webtoon text retains page-scale detail.
    if u64::from(height) * 2 <= u64::from(width) * 7 {
        return infer(image);
    }

    let crop_height = width.saturating_mul(2).min(height);
    let step = crop_height - (crop_height / 5).max(1);
    let mut detections: Vec<(KoharuLayoutDetection, (u32, u32))> = Vec::new();
    let mut top = 0;
    loop {
        let range = (top, top + crop_height);
        let crop = image.crop_imm(0, top, width, crop_height);
        for mut detection in infer(&crop)?.detections {
            detection.bbox[1] += top as f32;
            detection.bbox[3] += top as f32;
            detection.mask.y += top;
            if let Some((previous, previous_range)) =
                detections.iter_mut().find(|(previous, previous_range)| {
                    previous_range.0 != top
                        && same_instance(previous, &detection, *previous_range, range, height)
                })
            {
                merge(previous, detection);
                *previous_range = range;
            } else {
                detections.push((detection, range));
            }
        }
        if top == height - crop_height {
            break;
        }
        top = (top + step).min(height - crop_height);
    }
    Ok(KoharuLayoutDetections {
        image_width: width,
        image_height: height,
        detections: detections
            .into_iter()
            .map(|(detection, _)| detection)
            .collect(),
    })
}

fn same_instance(
    left: &KoharuLayoutDetection,
    right: &KoharuLayoutDetection,
    left_crop: (u32, u32),
    right_crop: (u32, u32),
    height: u32,
) -> bool {
    let top = left_crop.0.max(right_crop.0);
    let bottom = left_crop.1.min(right_crop.1);
    let intersection = (left.bbox[2].min(right.bbox[2]) - left.bbox[0].max(right.bbox[0])).max(0.0)
        * (left.bbox[3].min(right.bbox[3]) - left.bbox[1].max(right.bbox[1])).max(0.0);
    if left.label != right.label || top >= bottom || intersection <= 0.0 {
        return false;
    }
    let area = |bbox: [f32; 4]| (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0);
    let clipped = |detection: &KoharuLayoutDetection, crop: (u32, u32)| {
        (crop.0 > 0 && detection.bbox[1] <= crop.0 as f32 + 1.0)
            || (crop.1 < height && detection.bbox[3] >= crop.1 as f32 - 1.0)
    };
    if intersection / (area(left.bbox) + area(right.bbox) - intersection) < 0.5
        && !clipped(left, left_crop)
        && !clipped(right, right_crop)
    {
        return false;
    }
    // Compare only pixels visible to both crops. Whole-box IoU loses large
    // regions cut by a seam, while proximity can join unrelated text/panels.
    let band_area = |mask: &KoharuLayoutMask| {
        let start = top.saturating_sub(mask.y).min(mask.height) as usize * mask.width as usize;
        let end = bottom.saturating_sub(mask.y).min(mask.height) as usize * mask.width as usize;
        mask.pixels[start..end]
            .iter()
            .filter(|&&pixel| pixel != 0)
            .count()
    };
    let maximum = band_area(&left.mask).max(band_area(&right.mask));
    if maximum == 0 || left.mask.width == 0 {
        return false;
    }
    let mut shared = 0;
    for (index, &pixel) in left.mask.pixels.iter().enumerate() {
        let y = left.mask.y + (index / left.mask.width as usize) as u32;
        if pixel != 0 && y >= top && y < bottom {
            let x = left.mask.x + (index % left.mask.width as usize) as u32;
            shared += usize::from(right.mask.contains(x, y));
        }
    }
    shared as f64 / maximum as f64 >= 0.5
}

fn merge(left: &mut KoharuLayoutDetection, right: KoharuLayoutDetection) {
    let x = left.mask.x.min(right.mask.x);
    let y = left.mask.y.min(right.mask.y);
    let width = (left.mask.x + left.mask.width).max(right.mask.x + right.mask.width) - x;
    let height = (left.mask.y + left.mask.height).max(right.mask.y + right.mask.height) - y;
    let mut pixels = vec![0; width as usize * height as usize];
    for mask in [&left.mask, &right.mask] {
        for (index, &pixel) in mask.pixels.iter().enumerate() {
            let target_y = mask.y - y + (index / mask.width as usize) as u32;
            let target_x = mask.x - x + (index % mask.width as usize) as u32;
            let target = &mut pixels[target_y as usize * width as usize + target_x as usize];
            *target = (*target).max(pixel);
        }
    }
    left.bbox = [
        left.bbox[0].min(right.bbox[0]),
        left.bbox[1].min(right.bbox[1]),
        left.bbox[2].max(right.bbox[2]),
        left.bbox[3].max(right.bbox[3]),
    ];
    left.score = left.score.max(right.score);
    left.area = pixels.iter().filter(|&&pixel| pixel != 0).count() as u32;
    left.mask = KoharuLayoutMask {
        x,
        y,
        width,
        height,
        pixels,
    };
}

#[cfg(test)]
mod tests {
    use image::{GenericImageView, Rgb, RgbImage};

    use super::*;
    use crate::koharu_layout_rfdetr_seg_2xl::{KoharuLayoutDetection, KoharuLayoutMask};

    fn detection(label: &str, top: u32, bottom: u32) -> KoharuLayoutDetection {
        KoharuLayoutDetection {
            label_id: 0,
            label: label.to_owned(),
            score: 0.9,
            bbox: [1.0, top as f32, 3.0, bottom as f32],
            area: 2 * (bottom - top),
            mask: KoharuLayoutMask {
                x: 1,
                y: top,
                width: 2,
                height: bottom - top,
                pixels: vec![255; (2 * (bottom - top)) as usize],
            },
        }
    }

    #[test]
    fn ordinary_pages_keep_the_original_inference_path() {
        let image = DynamicImage::new_rgb8(100, 350);
        let mut calls = 0;
        let result = inference(&image, |input| {
            calls += 1;
            assert!(std::ptr::eq(input, &image));
            Ok(KoharuLayoutDetections {
                image_width: input.width(),
                image_height: input.height(),
                detections: Vec::new(),
            })
        })
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!((result.image_width, result.image_height), (100, 350));
    }

    #[test]
    fn overlapping_crops_preserve_source_pixels_and_restore_each_region_once() {
        for (text_top, text_bottom) in [(14, 22), (6, 30)] {
            let image =
                DynamicImage::ImageRgb8(RgbImage::from_fn(10, 45, |_, y| Rgb([y as u8, 0, 0])));
            let mut starts = Vec::new();
            let regions = [
                ("text", text_top, text_bottom),
                ("bubble", 27, 31),
                ("panel", 42, 45),
            ];
            let result = inference(&image, |input| {
                let top = u32::from(input.get_pixel(0, 0).0[0]);
                starts.push(top);
                assert_eq!((input.width(), input.height()), (10, 20));
                assert_eq!(input.get_pixel(0, 19).0[0], (top + 19) as u8);
                let detections = regions
                    .into_iter()
                    .filter_map(|(label, start, end)| {
                        let start = start.max(top);
                        let end = end.min(top + input.height());
                        (start < end).then(|| detection(label, start - top, end - top))
                    })
                    .collect();
                Ok(KoharuLayoutDetections {
                    image_width: input.width(),
                    image_height: input.height(),
                    detections,
                })
            })
            .unwrap();
            assert_eq!(starts, [0, 16, 25]);
            assert_eq!((result.image_width, result.image_height), (10, 45));
            assert_eq!(result.detections.len(), 3);
            for (detection, (label, top, bottom)) in result.detections.iter().zip(regions) {
                assert_eq!(detection.label, label);
                assert_eq!(detection.bbox, [1.0, top as f32, 3.0, bottom as f32]);
                assert_eq!(detection.mask.y, top);
                assert_eq!(detection.mask.height, bottom - top);
                assert_eq!(detection.area, 2 * (bottom - top));
                assert_eq!(
                    detection.mask.pixels,
                    vec![255; (2 * (bottom - top)) as usize]
                );
            }
        }
    }

    #[test]
    fn regions_detected_by_only_one_overlap_crop_are_retained() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(10, 45, |_, y| Rgb([y as u8, 0, 0])));
        for (crop, top, bottom) in [(0, 18, 20), (16, 16, 18)] {
            let result = inference(&image, |input| {
                let crop_top = u32::from(input.get_pixel(0, 0).0[0]);
                Ok(KoharuLayoutDetections {
                    image_width: input.width(),
                    image_height: input.height(),
                    detections: if crop_top == crop {
                        vec![detection("text", top - crop, bottom - crop)]
                    } else {
                        Vec::new()
                    },
                })
            })
            .unwrap();
            assert_eq!(result.detections.len(), 1);
            assert_eq!(
                result.detections[0].bbox,
                [1.0, top as f32, 3.0, bottom as f32]
            );
            assert_eq!(result.detections[0].mask.y, top);
        }
    }

    #[test]
    fn nearby_distinct_masks_and_different_labels_are_not_merged() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(10, 45, |_, y| Rgb([y as u8, 0, 0])));
        for distinct_masks in [true, false] {
            let result = inference(&image, |input| {
                let top = u32::from(input.get_pixel(0, 0).0[0]);
                let detections = if top == 0 || top == 16 {
                    let mut region = if top == 0 {
                        detection("text", 14, 20)
                    } else {
                        detection(if distinct_masks { "text" } else { "bubble" }, 0, 6)
                    };
                    if distinct_masks {
                        for row in region.mask.pixels.chunks_exact_mut(2) {
                            row[usize::from(top == 0)] = 0;
                        }
                        region.area = 6;
                    }
                    vec![region]
                } else {
                    Vec::new()
                };
                Ok(KoharuLayoutDetections {
                    image_width: input.width(),
                    image_height: input.height(),
                    detections,
                })
            })
            .unwrap();
            assert_eq!(result.detections.len(), 2);
        }
    }

    #[test]
    fn empty_images_and_inference_failures_return_errors() {
        assert!(inference(&DynamicImage::new_rgb8(0, 1), |_| panic!("no crop")).is_err());
        let image = DynamicImage::new_rgb8(10, 45);
        let result = inference(&image, |_| anyhow::bail!("inference failed"));
        assert_eq!(result.unwrap_err().to_string(), "inference failed");
    }
}
