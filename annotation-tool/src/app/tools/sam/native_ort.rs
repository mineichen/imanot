use std::{future::Future, path::Path, sync::Arc};

use imask::NonZeroRange;
use log::debug;
use ndarray::Array;
use ort::{Error as OrtError, session::Session, value::Value};

use super::{InferenceError, RgbImageInterleaved, SamEmbeddings, inference::extract_pixel_ranges};

#[derive(Clone)]
pub struct SamSession {
    encoder: Arc<std::sync::Mutex<Session>>,
    decoder: Arc<std::sync::Mutex<Session>>,
}

impl SamSession {
    pub fn new(path: &Path) -> Result<Self, InferenceError> {
        eprintln!("[DEBUG] SamSession::new called with path: {:?}", path);
        eprintln!("[DEBUG] About to create Session::builder()...");
        let mut builder = Session::builder()?;
        eprintln!("[DEBUG] Session::builder() succeeded");
        let encoder = builder.commit_from_file(path.join("vit_t_encoder.onnx"))?;
        eprintln!("[DEBUG] encoder loaded");
        let decoder = Session::builder()?.commit_from_file(path.join("vit_t_decoder.onnx"))?;
        eprintln!("[DEBUG] decoder loaded");

        Ok(Self {
            encoder: Arc::new(encoder.into()),
            decoder: Arc::new(decoder.into()),
        })
    }

    pub fn get_image_embeddings(
        &self,
        img: RgbImageInterleaved<u8>,
    ) -> impl Future<Output = Result<SamEmbeddings, InferenceError>> + Send + 'static {
        let (tx, rx) = futures::channel::oneshot::channel();

        let session = self.encoder.clone();
        let handle = std::thread::spawn(move || {
            let mut session = session.lock().expect("Mutex is not poisoned");
            let r = Self::get_image_embeddings_blocking(&mut session, img);
            tx.send(r)
        });
        async move {
            let r = rx
                .await
                .map_err(|e| InferenceError::Other(Arc::new(e)))
                .and_then(|a| a);
            handle
                .join()
                .expect("SAM thread does not panic")
                .expect("Channel cant be gone");
            r
        }
    }
    pub fn get_image_embeddings_blocking(
        encoder: &mut Session,
        img: RgbImageInterleaved<u8>,
    ) -> Result<SamEmbeddings, InferenceError> {
        let image_input = super::inference::prepare_image_input(&img)?;
        // Prepare tensor for the SAM encoder model
        let input_as_values = image_input.image_data.to_owned();
        let encoder_inputs = ort::inputs![Value::from_array(input_as_values)?];

        // Run encoder to get image embeddings
        let outputs = encoder.run(encoder_inputs)?;
        // return Err(InferenceError::Other(Arc::new(std::io::Error::other(
        //     "Testing purpose",
        // ))));
        let embeddings = outputs
            .into_iter()
            .next()
            .ok_or_else(|| InferenceError::UnexpectedOutput("Expected a output".into()))?
            .1
            .try_extract_array::<f32>()
            .map_err(|e| InferenceError::UnexpectedOutput(format!("Expected f32: {e:?}")))?
            .view()
            .t()
            .reversed_axes()
            .into_owned();

        let embeddings = Value::from_array(embeddings)?;
        Ok(image_input.map(|_| embeddings))
    }

    pub fn decode_prompt(
        &self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        embeddings: &SamEmbeddings,
    ) -> Result<Vec<NonZeroRange<u64>>, InferenceError> {
        // Prepare input for decoder

        // Get embeddings, image sizes and ONNX model instances from Web Application state
        let orig_width = embeddings.original_width.get() as f32;
        let orig_height = embeddings.original_height.get() as f32;
        let resized_width = embeddings.resized_width.get() as f32;
        let resized_height = embeddings.resized_height.get() as f32;
        let mut decoder = self.decoder.lock().expect("Mutex is not poisoned");
        let embeddings_as_values = embeddings.image_data.clone();

        let x_ratio = resized_width / orig_width;
        let y_ratio = resized_height / orig_height;
        let point_coords =
            ndarray::array![[[x1 * x_ratio, y1 * y_ratio], [x2 * x_ratio, y2 * y_ratio]]];

        // Labels
        let point_labels = ndarray::array![[2.0_f32, 3.0_f32]];

        // Encode mask prompt (dummy)
        let mask_input = Array::<f32, _>::zeros((1, 1, 256, 256));
        let has_mask_input = ndarray::array![0.0_f32];

        // Add original image size
        let orig_im_size = ndarray::array![orig_height, orig_width];

        // Prepare inputs for SAM decoder
        let decoder_inputs = ort::inputs![
            embeddings_as_values,
            Value::from_array(point_coords)?,
            Value::from_array(point_labels)?,
            Value::from_array(mask_input)?,
            Value::from_array(has_mask_input)?,
            Value::from_array(orig_im_size)?,
        ];

        // Run the SAM decoder
        let outputs = decoder.run(decoder_inputs)?;
        debug!(
            "Outputs {:?}",
            outputs
                .iter()
                .map(|(_, x)| x.try_extract_array::<f32>().map(|x| x.view().len()))
                .collect::<Vec<_>>()
        );

        // Process and return output mask (replace negative pixel values to 0 and positive to 1)
        let pixels = outputs
            .into_iter()
            .next()
            .ok_or_else(|| InferenceError::UnexpectedOutput("No output".into()))?
            .1;
        let pixels = pixels
            .try_extract_array::<f32>()
            .map_err(|e| InferenceError::UnexpectedOutput(format!("Output of type f32: {e:?}")))?;
        let pixel_view = pixels.view();

        Ok(extract_pixel_ranges(
            pixel_view.iter().copied(),
            embeddings.original_width,
        ))
    }
}

impl From<OrtError> for InferenceError {
    fn from(value: OrtError) -> Self {
        InferenceError::Other(Arc::new(value))
    }
}
