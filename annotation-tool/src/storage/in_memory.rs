use std::{
    collections::HashMap,
    io,
    sync::{Arc, Mutex},
};

use futures::{FutureExt, future::BoxFuture};
use imanot::PixelAreaStack;

use super::{ImageData, ImageId, ImageListTaskItem, Storage};

pub struct InMemoryStorage {
    data: Arc<Mutex<HashMap<ImageId, ImageData>>>,
}

impl InMemoryStorage {
    pub fn new(x: impl IntoIterator<Item = ImageData>) -> Self {
        Self {
            data: Arc::new(Mutex::new(
                x.into_iter().map(|x| (x.id.clone(), x)).collect(),
            )),
        }
    }

    pub fn chessboard() -> Self {
        Self::new(ImageData::chessboard())
    }
}

impl Storage for InMemoryStorage {
    fn list_images(&self) -> BoxFuture<'static, io::Result<Vec<ImageListTaskItem>>> {
        let data = self.data.lock().expect("Mutex is not poisoned");
        let result = data
            .iter()
            .map(|(id, data)| {
                let name = id
                    .chars()
                    .enumerate()
                    .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
                    .collect::<String>();
                ImageListTaskItem {
                    id: id.clone(),
                    name,
                    has_masks: !data.masks.is_empty(),
                }
            })
            .collect::<Vec<_>>();
        std::future::ready(Ok(result)).boxed()
    }

    fn load_image(&self, id: &ImageId) -> BoxFuture<'static, io::Result<ImageData>> {
        let id = id.clone();
        let data = self
            .data
            .lock()
            .expect("Mutex is not poisoned")
            .get(&id)
            .cloned()
            .ok_or_else(|| std::io::Error::other(format!("Unknown image_id {id:?}")));
        std::future::ready(data).boxed()
    }

    fn store_masks(
        &self,
        id: &ImageId,
        masks: &PixelAreaStack,
    ) -> BoxFuture<'static, io::Result<()>> {
        if let Some(x) = self.data.lock().expect("Mutex is not poisoned").get_mut(id) {
            x.masks = masks.clone();
        };
        async move {
            // Implement storing masks to web storage or IndexedDB
            Ok(())
        }
        .boxed()
    }
}
