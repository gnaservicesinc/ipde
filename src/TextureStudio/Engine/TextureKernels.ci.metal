#include <metal_stdlib>
#include <CoreImage/CoreImage.h>
using namespace metal;

extern "C" { namespace coreimage {
    float textureLuminance(float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }

    float4 textureDelight(sample_t photo, sample_t lighting, float target, float strength, float hdrInput) {
        float illumination = max(textureLuminance(lighting.rgb / max(lighting.a, 1e-6f)), 0.005f);
        float gain = clamp(target / illumination, 0.4f, 2.5f);
        // Color transforms work on straight color, while Core Image samples and
        // returns premultiplied color. Source opacity is independent of crop.
        float3 color = photo.rgb / max(photo.a, 1e-6f);
        float3 result = max(color * pow(gain, strength), 0.0f);
        if (hdrInput > 0.5f) {
            float peak = max(result.r, max(result.g, result.b));
            if (peak > 0.85f) {
                // Gentle shared-channel shoulder keeps HDR surface colour ratios and avoids
                // turning recovered gain-map highlights into a hard white clipping plateau.
                float shoulder = 0.85f + 0.15f * (1.0f - exp(-(peak - 0.85f) / 0.15f));
                result *= shoulder / peak;
            }
        }
        result = clamp(result, 0.0f, 1.0f);
        return float4(result * photo.a, photo.a);
    }

    float4 textureHeight(sample_t baseHeight, sample_t photo, sample_t lowPhoto, float photoStrength) {
        // Apple's fast Gaussian can quantize intermediate samples even in a float context.
        // Unpremultiply its weight sum and suppress its sub-0.05% numerical contrast floor
        // so a flat photograph does not acquire false bumps or roughness stripes.
        float contrast = photo.a > 1e-6f ? textureLuminance(photo.rgb / photo.a)
            - textureLuminance(lowPhoto.rgb / max(lowPhoto.a, 1e-6f)) : 0.0f;
        float detail = sign(contrast) * max(abs(contrast) - 0.0005f, 0.0f) * 3.0f;
        // The registered depth establishes all geometry by default. Photo contrast
        // can only add explicitly requested artistic relief; colour is not height.
        float h = clamp(baseHeight.r + detail * photoStrength, 0.0f, 1.0f);
        return float4(h, h, h, 1.0f);
    }

    float4 textureRoughness(sample_t photo, sample_t lowPhoto, float base, float detailStrength) {
        // An editable material estimate; a single colour photograph cannot measure roughness.
        float contrast = photo.a > 1e-6f ? textureLuminance(photo.rgb / photo.a)
            - textureLuminance(lowPhoto.rgb / max(lowPhoto.a, 1e-6f)) : 0.0f;
        float localContrast = max(abs(contrast) - 0.0005f, 0.0f);
        float roughness = clamp(base + localContrast * detailStrength * 4.0f, 0.02f, 1.0f);
        return float4(roughness, roughness, roughness, 1.0f);
    }

    float4 textureNormal(sampler height, float physicalSlope, destination destination) {
        float2 p = destination.coord();
        float left = height.sample(height.transform(p - float2(1.0f, 0.0f))).r;
        float right = height.sample(height.transform(p + float2(1.0f, 0.0f))).r;
        float bottom = height.sample(height.transform(p - float2(0.0f, 1.0f))).r;
        float top = height.sample(height.transform(p + float2(0.0f, 1.0f))).r;
        float3 normal = normalize(float3(-(right-left)*physicalSlope,
                                        -(top-bottom)*physicalSlope, 1.0f));
        return float4(normal * 0.5f + 0.5f, 1.0f);
    }

    float2 textureLens(float2 centre, float2 halfExtent, float distortion, float zoom,
                       destination destination) {
        float2 delta = (destination.coord() - centre) / zoom;
        float2 normalised = delta / halfExtent;
        return centre + delta * (1.0f + distortion * dot(normalised, normalised));
    }
}}
