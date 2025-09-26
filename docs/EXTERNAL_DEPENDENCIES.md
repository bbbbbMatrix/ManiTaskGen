# External Dependencies

## VLMEvalKit

- **Source**: [https://github.com/open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- **Location**: `src/vlm_interaction/VLMEvalKit`
- **Status**: Contains custom modifications for project-specific requirements

### Important Notes
- This directory contains modified code from the original VLMEvalKit project
- It is NOT a standard VLMEvalKit distribution
- Custom modifications have been applied to integrate with our task generation system
- Please handle updates carefully to avoid overwriting custom changes

### Installation Instructions
If you need to reinstall from the original source:
```bash
# Clone the original repository
git clone https://github.com/open-compass/VLMEvalKit.git temp_vlmkit

# Apply necessary custom modifications manually
# Refer to git history for specific changes made
```

### Custom Modifications
- Integration with task generation pipeline
- Custom configuration handling
- Modified evaluation interfaces

For detailed modification history, please check the git commit history of this repository.
