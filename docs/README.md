# Texture Studio documentation

Texture Studio and its material tools are native SwiftUI macOS applications. Building requires Xcode and the Apple SDK; no separate interpreter or package environment is used.

- [Surface and Blender workflow](texture-studio.md)
- [Native dataset, review and training tools](native-material-tools.md)
- [Training and precision contract](material-training-data.md)
- [Training workflow](material-training-workflow.md)
- [Model setup](model-setup.md)
- [Training configuration and upstream model audit](training-audit.md)
- [Building and releasing](releasing.md)
- [Native migration validation](native-migration-validation.md)
- [Preparation performance and training crash validation](performance-validation.md)

The app uses ImageIO for image and auxiliary-data access, Core Image and Metal for material processing, Accelerate for numeric statistics, MPSGraph for material inference and adapter training, and URLSession for model transfers. Original photos and numeric source maps remain separate from previews and model inputs.
