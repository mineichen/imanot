use std::{
    fs::DirEntry,
    io::{self, ErrorKind, Read, Write},
    num::{NonZero, NonZeroU32},
    ops::Range,
    path::PathBuf,
    str::FromStr,
};

use bytemuck::{AnyBitPattern, NoUninit};
use futures::{FutureExt, future::BoxFuture};
use imanot::{ImageData, ImageId, PixelArea, PixelAreaStack, load_image};
use imask::{ImageDimension, ImaskSet, Rect, Roi, SignedNonZeroable, Span};
use itertools::Itertools;
use log::info;

use super::{ImageListTaskItem, Kind, MaybeOneOrMany, PREAMBLE, Storage, VERSION};

pub struct FileStorage {
    base: String,
}
impl FileStorage {
    pub fn new(base: impl Into<String>) -> Self {
        Self { base: base.into() }
    }

    fn list_images_blocking(path: PathBuf) -> std::io::Result<Vec<ImageListTaskItem>> {
        Ok(visit_directory_files(path)
            .filter_map(|x| {
                let x = x.ok()?;
                let path = x.path();
                let kind = path
                    .extension()?
                    .to_str()
                    .and_then(|s| Kind::from_str(s).ok())?;
                Some((
                    path.file_stem()
                        .expect("exists_if_extension_exists")
                        .to_string_lossy()
                        .to_string(),
                    kind,
                    path.to_str()?.into(),
                ))
            })
            .sorted_unstable()
            .chunk_by(|x| x.0.to_string()) // Pitty...
            .into_iter()
            .filter_map(|(_, mut members)| {
                let (name, kind, id) = members.next().expect("Needs one item to form a group");
                match (kind, members.next()) {
                    (Kind::Mask, None) => None,
                    (Kind::Mask, Some((_, Kind::Mask, _))) => {
                        unreachable!("Cannot have multiple file_stem.mask")
                    }
                    // Takeing any image is fine, ignore the rest
                    (Kind::Mask, Some((name, Kind::Image, id))) => Some(ImageListTaskItem {
                        id,
                        name,
                        has_masks: true,
                    }),
                    (Kind::Image, _) => Some(ImageListTaskItem {
                        id,
                        name,
                        has_masks: false,
                    }),
                }
            })
            .collect::<Vec<_>>())
    }
    fn get_image_path(&self) -> PathBuf {
        self.base.as_str().into()
    }

    fn get_mask_path(id: &ImageId) -> std::io::Result<PathBuf> {
        let file_path = std::path::Path::new(&**id);

        let filename = file_path
            .file_stem()
            .and_then(|x| x.to_str())
            .ok_or_else(|| std::io::Error::other("File has no filename"))?;
        let images_path = file_path
            .parent()
            .ok_or_else(|| std::io::Error::other("Base musten't be a root-dir"))?;

        Ok(images_path.join(format!("{filename}.masks")))
    }
}

impl Storage for FileStorage {
    // uri -> Display
    fn list_images(&self) -> BoxFuture<'static, std::io::Result<Vec<ImageListTaskItem>>> {
        let (tx, rx) = futures::channel::oneshot::channel();
        let image_path = self.get_image_path();

        let handle = std::thread::spawn(|| {
            let r = Self::list_images_blocking(image_path);
            tx.send(r)
        });
        async move {
            let r = rx.await.map_err(std::io::Error::other).and_then(|a| a);
            handle.join().unwrap().expect("Channel cant be gone");
            r
        }
        .boxed()
    }

    fn load_image(&self, id: &ImageId) -> BoxFuture<'static, std::io::Result<ImageData>> {
        fn load_internal<
            TLen: SignedNonZeroable + Copy + NoUninit + AnyBitPattern + Default + std::fmt::Debug,
        >(
            f: std::fs::File,
            image_width: NonZeroU32,
            image_height: NonZeroU32,
            file_version: u16,
        ) -> io::Result<Vec<PixelArea>>
        where
            u32: From<TLen>,
        {
            let mut f = brotli::Decompressor::new(f, 4096);
            let mut pixel_range_bytes = [0; 2];
            let mut all = Vec::new();
            let mut starts = Vec::<u32>::new();
            let mut lens = Vec::<TLen>::new();

            fn read_u32<T: Read>(mut r: T) -> io::Result<u32> {
                let mut buf = [0; 4];
                r.read_exact(&mut buf)?;
                Ok(u32::from_le_bytes(buf))
            }
            fn read_nz_u32<T: Read>(r: T, reason: &str) -> io::Result<NonZero<u32>> {
                let data = read_u32(r)?;
                NonZero::new(data).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, reason))
            }

            while f.read_exact(&mut pixel_range_bytes).is_ok() {
                let pixel_range_len = u16::from_le_bytes(pixel_range_bytes) as usize;
                if pixel_range_len == 0 {
                    continue;
                }
                let (bounds, color) = if file_version == 1 {
                    let color = imanot::random_color_from_seed(all.len() as u16);
                    (
                        Roi::new_unchecked(0u32..image_width.get(), 0..image_height.get()),
                        color,
                    )
                } else {
                    let offset_x = read_u32(&mut f)?;
                    let offset_y = read_u32(&mut f)?;
                    let width = read_nz_u32(&mut f, "NonZero width")?;
                    let height = read_nz_u32(&mut f, "NonZero height")?;
                    let x_end = offset_x.checked_add(width.get()).ok_or(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "x + width overflows",
                    ))?;

                    let y_end = offset_y.checked_add(height.get()).ok_or(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "y + height overflows",
                    ))?;

                    let color = imanot::random_color_from_seed(all.len() as u16);
                    (Roi::new(offset_x..x_end, offset_y..y_end), color)
                };

                starts.resize(pixel_range_len, 0);
                lens.resize(pixel_range_len, Default::default());
                f.read_exact(bytemuck::cast_slice_mut(&mut starts))?;
                f.read_exact(bytemuck::cast_slice_mut(&mut lens))?;
                // Generate color based on current position (simulating the seed)
                let pixels = starts
                    .iter()
                    .zip(lens.iter())
                    .map(|(start, len)| match TLen::create_non_zero(*len) {
                        Some(l) => {
                            let x = *start % bounds.width().get() + bounds.x.start;
                            let y = *start / bounds.width().get() + bounds.y.start;
                            Ok(Span::new(x..x + u32::from(l.into()), y))
                        }
                        None => Err(std::io::Error::new(
                            ErrorKind::InvalidData,
                            format!("position {start},{len:?}: Found ZeroValue"),
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .with_roi(bounds);
                let area =
                    PixelArea::new(pixels, color).expect("Group cannot be empty, checked in loop");
                all.push(area);
            }

            Ok(all)
        }

        let id = id.clone();
        async move {
            let image_bytes = std::fs::read(&*id)?;
            let mask_path = Self::get_mask_path(&id)?;

            let image_load_ok = load_image(&image_bytes)?;
            let image_width = image_load_ok.original.width();
            let image_height = image_load_ok.original.height();
            let masks = match std::fs::File::open(mask_path) {
                Ok(mut f) => {
                    let mut preamble = [0; PREAMBLE.len()];
                    f.read_exact(&mut preamble)?;
                    if preamble != PREAMBLE {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Invalid preamble",
                        ));
                    }
                    let mut version_bytes = [0; 2];
                    f.read_exact(&mut version_bytes)?;
                    let file_version = u16::from_le_bytes(version_bytes);
                    match file_version {
                        1 => load_internal::<u16>(f, image_width, image_height, file_version)?,
                        2 => load_internal::<u32>(f, image_width, image_height, file_version)?,
                        x => {
                            return Err(io::Error::new(
                                io::ErrorKind::Unsupported,
                                format!("Unsupported version {x}"),
                            ));
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Default::default(),
                Err(e) => return Err(e),
            };
            Ok(ImageData::new(
                id,
                image_load_ok,
                PixelAreaStack::from(masks),
                imanot::HistoryStrategy::Reset,
            ))
        }
        .boxed()
    }

    fn store_masks(
        &self,
        id: &ImageId,
        masks: &PixelAreaStack,
    ) -> BoxFuture<'static, io::Result<()>> {
        let path = Self::get_mask_path(id);
        let masks = masks.clone();
        async move {
            info!("Store at: {path:?}");
            let path = path?;
            if masks.is_empty() {
                match std::fs::remove_file(path) {
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e),
                }
            } else {
                let mut f = std::fs::File::create(path)?;
                f.write_all(&PREAMBLE)?;
                f.write_all(&VERSION.to_le_bytes())?;

                let mut f = brotli::CompressorWriter::new(f, 4096, 11, 22);
                for (_layer, sub) in masks.iter() {
                    if sub.range_len() > u16::MAX as _ {
                        return Err(std::io::Error::new(
                            ErrorKind::InvalidData,
                            format!(
                                "Version1 allows for MAX {} subgroups, got {}",
                                u16::MAX,
                                sub.range_len()
                            ),
                        ));
                    }
                    let sub_len = sub.range_len() as u16;

                    f.write_all(&sub_len.to_le_bytes())?;
                    let bounds = sub.pixels.bounds();
                    f.write_all(&bounds.x.to_le_bytes())?;
                    f.write_all(&bounds.y.to_le_bytes())?;
                    f.write_all(&bounds.width.get().to_le_bytes())?;
                    f.write_all(&bounds.height.get().to_le_bytes())?;
                    for subgroup in sub.pixels.iter_roi::<Range<u32>>() {
                        f.write_all(&subgroup.start.to_le_bytes())?;
                    }
                    for subgroup in sub.pixels.iter_roi::<Range<u32>>() {
                        f.write_all(&u32::try_from(subgroup.len()).unwrap().to_le_bytes())?;
                    }
                }

                f.flush()?;
            }
            Ok(())
        }
        .boxed()
    }
}

pub fn visit_directory_files(
    path: impl Into<PathBuf>,
) -> impl Iterator<Item = std::io::Result<DirEntry>> {
    fn one_level(path: PathBuf) -> MaybeOneOrMany<std::io::Result<DirEntry>> {
        match std::fs::read_dir(path) {
            Ok(readdir) => MaybeOneOrMany::Many(Box::new(readdir.flat_map(|entry| match entry {
                Ok(entry) => match entry.file_type() {
                    Ok(filetype) => {
                        if filetype.is_dir() {
                            one_level(entry.path())
                        } else {
                            MaybeOneOrMany::MaybeOne(Some(Ok(entry)))
                        }
                    }
                    Err(e) => MaybeOneOrMany::MaybeOne(Some(Err(e))),
                },
                Err(e) => MaybeOneOrMany::MaybeOne(Some(Err(e))),
            }))),
            Err(e) => MaybeOneOrMany::MaybeOne(Some(Err(e))),
        }
    }
    one_level(path.into())
}
