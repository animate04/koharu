use std::{io::Cursor, path::Path, sync::Arc};

use anyhow::{Context as _, Result};
use image::{DynamicImage, GenericImageView, ImageFormat};

use super::Page;

// Keep imported strips within the baseline WebGPU texture height for editing and export.
const MAX_HEIGHT: u32 = 8192;
const GUTTER_SEARCH: u32 = 1024;
const GUTTER_ROWS: u32 = 16;

pub(super) fn split(page: Page) -> Result<Vec<Page>> {
    if page.height <= MAX_HEIGHT {
        return Ok(vec![page]);
    }
    let image = image::load_from_memory_with_format(&page.bytes, page.format)
        .with_context(|| format!("failed to decode long page {}", page.name))?;
    let mut pages = Vec::new();
    let mut top = 0;
    while top < page.height {
        let end = top.saturating_add(MAX_HEIGHT).min(page.height);
        let bottom = if end == page.height {
            end
        } else {
            gutter(&image, end - GUTTER_SEARCH, end).unwrap_or(end)
        };
        let height = bottom - top;
        let part = image.crop_imm(0, top, page.width, height);
        let mut bytes = Cursor::new(Vec::new());
        part.write_to(&mut bytes, ImageFormat::Png)
            .with_context(|| format!("failed to encode long page {}", page.name))?;
        pages.push(Page {
            name: format!(
                "{}-part-{:03}.png",
                Path::new(&page.name).with_extension("").display(),
                pages.len() + 1
            ),
            bytes: Arc::from(bytes.into_inner()),
            format: ImageFormat::Png,
            width: page.width,
            height,
        });
        top = bottom;
    }
    Ok(pages)
}

fn gutter(image: &DynamicImage, start: u32, end: u32) -> Option<u32> {
    let mut run = 0;
    let mut cut = None;
    for y in start..end {
        let color = image.get_pixel(0, y);
        let uniform = (0..image.width()).all(|x| {
            image
                .get_pixel(x, y)
                .0
                .into_iter()
                .zip(color.0)
                .all(|(a, b)| a.abs_diff(b) <= 8)
        });
        run = if uniform { run + 1 } else { 0 };
        if run >= GUTTER_ROWS {
            cut = Some(y + 1 - GUTTER_ROWS / 2);
        }
    }
    // shortcut: without a gutter a panel can be divided; add panel-aware cuts if whole panels are required.
    cut
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{RgbaImage, imageops};

    fn page(image: impl Into<DynamicImage>) -> Page {
        let image = image.into();
        let (width, height) = image.dimensions();
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, ImageFormat::Png).unwrap();
        Page {
            name: "chapter.png".into(),
            bytes: Arc::from(bytes.into_inner()),
            format: ImageFormat::Png,
            width,
            height,
        }
    }

    #[test]
    fn short_pages_keep_their_original_bytes_and_name() {
        let original = page(RgbaImage::new(2, MAX_HEIGHT));
        let bytes = original.bytes.clone();
        let pages = split(original).unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].name, "chapter.png");
        assert!(Arc::ptr_eq(&pages[0].bytes, &bytes));
    }

    #[test]
    fn long_strips_split_at_gutters_without_losing_or_repeating_pixels() {
        let mut image = RgbaImage::from_fn(2, MAX_HEIGHT * 2 + 35, |x, y| {
            image::Rgba([if x == 0 { 0 } else { 255 }, (y % 251) as u8, 40, 255])
        });
        for y in MAX_HEIGHT - 100..MAX_HEIGHT - 60 {
            for x in 0..image.width() {
                image.put_pixel(x, y, image::Rgba([255; 4]));
            }
        }
        let pages = split(page(image.clone())).unwrap();
        assert_eq!(pages.len(), 3);
        assert!(pages[0].height < MAX_HEIGHT);
        let mut reconstructed = RgbaImage::new(image.width(), image.height());
        let mut top = 0;
        for (index, part) in pages.iter().enumerate() {
            assert!(part.height > 0 && part.height <= MAX_HEIGHT);
            assert_eq!(part.name, format!("chapter-part-{:03}.png", index + 1));
            let pixels = image::load_from_memory_with_format(&part.bytes, part.format)
                .unwrap()
                .into_rgba8();
            imageops::replace(&mut reconstructed, &pixels, 0, i64::from(top));
            top += part.height;
        }
        assert_eq!(top, image.height());
        assert_eq!(reconstructed, image);
    }

    #[test]
    fn long_png_parts_preserve_sixteen_bit_channels() {
        let image = image::ImageBuffer::from_fn(2, MAX_HEIGHT + 1, |x, y| {
            image::Rgba([
                if x == 0 { 1u16 } else { 65534 },
                (y.wrapping_mul(513) % 65536) as u16,
                12345,
                65535,
            ])
        });
        let pages = split(page(DynamicImage::ImageRgba16(image.clone()))).unwrap();
        let mut top = 0;
        for part in pages {
            let decoded = image::load_from_memory_with_format(&part.bytes, part.format).unwrap();
            assert_eq!(decoded.color(), image::ColorType::Rgba16);
            let pixels = decoded.into_rgba16();
            for (x, y, pixel) in pixels.enumerate_pixels() {
                assert_eq!(pixel, image.get_pixel(x, top + y));
            }
            top += part.height;
        }
        assert_eq!(top, image.height());
    }
}
