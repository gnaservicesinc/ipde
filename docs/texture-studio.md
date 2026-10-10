# Native Texture Studio workflow

Texture Studio is a macOS Swift app. Image import, auxiliary extraction, image processing, dataset management, material inference, training, model transfers and exports run through native services. The app uses ImageIO, Core Image, Accelerate, Metal and Metal Performance Shaders; it does not launch an interpreter.

## Photograph and material preparation

ImageIO supplies the primary photo, camera metadata, HDR gain maps and available auxiliary data. Perspective controls project a plane through a camera-centered pinhole model. Camera metadata can supply approximate focal length; a user override is available. The square crop remains inside the projected footprint. Nonplanar surfaces remain an approximation.

Photographic processing uses extended linear sRGB and a Metal Float32 context. HDR color is decoded before geometry and cropping. Broad illumination correction and a photographic highlight rolloff prepare the diffuse map. The finished diffuse pixels are shared by preview, export and model inference. Illumination correction cannot reconstruct fully hidden shadows or clipped highlights.

Numeric maps use a separate color-unmanaged Float32 context. Height, roughness and normals receive no photographic gamma or tone mapping. Registered surface height keeps its supplied amplitude through explicit relief controls. Height-derived normals use mathematical differences, material width and displacement amplitude, following OpenGL +Y. Embedded portrait scene depth is not material displacement; with no attached material height or selected material model, the default displacement is flat.

## Precision and export

Source bit depth, model arithmetic and container precision are separate. Diffuse export is 16-bit sRGB PNG. Numeric maps use HALF or FLOAT EXR according to the selected precision. The explicit FLOAT writer stores Float32 samples without range stretching or lossy compression. Changing output dimensions or container precision does not add source detail.

Original-map exports copy checksum-verified file bytes. Display contrast affects only previews. Native training-grid reconstruction reverses PNG compression, row filters and Adam7 directly on integer codes, then copies the exact recorded crop. It never applies a display color conversion to the source maps.

Material exports include `BLENDER.txt` and `material-nodes.json`. They record filenames, sRGB diffuse and Non-Color numeric maps, OpenGL +Y normals, material width and displacement scale in meters. Choose normal shading or geometric displacement; disable the equivalent height-derived normal contribution when geometry supplies that same relief.

## Datasets and training

Material Dataset creates named datasets, registers original diffuse and surface maps, scans import folders, edits notes and review status, removes membership and moves verified owned metadata to Trash. Original image files remain untouched. Folder import previews are bound to the dataset revision, review revision and original file identities. Another window changing any of these requires a new scan.

One native grid applies to diffuse variants, targets and learning-check crops. Exact-size maps are referenced directly. Larger sources use a centered integer crop; sources at least 8K on both axes use three corner crops. Sources below the selected size remain inspectable and are unavailable for that run. Crop coordinates match across roles and colors. Owned temporary training files live under `.training-data/`; cleanup retains source-bound reviews and refuses unrelated files.

The native PBRnxt graph uses its SCUNetV2 encoder, generator fusion and complete selected RRDB output branch. Native-grid adaptation retains the learned upscale convolutions and omits the final 4× image enlargement. Metal executes inference and gradients for LoRA refinement; Swift averages accumulated gradients and applies configurable Adam or AdamW updates to the compact factors. Final-map scope refines the selected output branch; map-decoder scope also refines the corresponding generator decoder and tail. Numeric training targets enter Float32 tensors as integer code divided by the declared type maximum. The model input color transfer is explicit and separate from source storage. See the [configuration audit](training-audit.md) for effective defaults and upstream differences.

Native checkpoints use `.safetensors` with recorded architecture, exact base checksum, target, scope and adapter shape. Normal mode packages the adapter; Developer mode can package fused full weights. Weighted adapter mixing requires matching base identity and module layout. Model downloads use recorded repository revisions. Upload visibility defaults to private, and automatic upload is shown before it is enabled.

Numeric fields accept direct typing and paste, state their units and reject invalid values. Resolution choices follow the model grids and original dimensions. The trainer names the selected dataset, material and starting checkpoint. Checkpoint actions allow saving at an update boundary, saving and stopping, or aborting. Model memory and machine headroom are reported independently of the selected image grid.

## Adviser and quality review

The optional local Clef adviser receives bounded display copies through Ollama. It returns typed recommendations for preparation or map review; the user chooses which to apply. Adviser display images never replace numerical originals, and model scores do not establish physical accuracy.

Compare reference, base and refined outputs on identical prepared diffuse pixels at the actual model grid. Inspect grain, relief placement, inversion, halos, seams, edge frames and displaced geometry under neutral and grazing light. Automatic learning checks use a different crop of a known subject and may share pixels with other training views. They do not measure unseen-material generalization.

Tiny native graph, gradient and checkpoint fixtures validate the implementation. Production-size published weights and visual parity require a separate run with the selected base and real reference materials. See the [data contract](material-training-data.md) and [native material tools](native-material-tools.md).
