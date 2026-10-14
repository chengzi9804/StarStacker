#include <metal_stdlib>
using namespace metal;

kernel void accumulateWithShift(
    texture2d<float, access::read_write> sumTex [[texture(0)]],
    texture2d<float, access::read_write> sumSqTex [[texture(1)]],
    texture2d<float, access::read_write> countTex [[texture(2)]],
    texture2d<float, access::read> frameTex [[texture(3)]],
    constant float2 &shift [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint w = sumTex.get_width();
    uint h = sumTex.get_height();

    int sx = int(gid.x) - int(round(shift.x));
    int sy = int(gid.y) - int(round(shift.y));

    float4 sum = sumTex.read(gid);
    float sumSq = sumSqTex.read(gid).r;
    float count = countTex.read(gid).r;

    if (sx >= 0 && sx < int(w) && sy >= 0 && sy < int(h)) {
        float4 pix = frameTex.read(uint2(sx, sy));
        float3 rgb = pix.rgb;
        float lum = (rgb.r + rgb.g + rgb.b) / 3.0;
        sum.rgb += rgb;
        sumSq += lum * lum;
        count += 1.0;
    }

    sumTex.write(sum, gid);
    sumSqTex.write(float4(sumSq, 0, 0, 1), gid);
    countTex.write(float4(count, 0, 0, 1), gid);
}

kernel void sigmaClipOutput(
    texture2d<float, access::read> sumTex [[texture(0)]],
    texture2d<float, access::read> sumSqTex [[texture(1)]],
    texture2d<float, access::read> countTex [[texture(2)]],
    texture2d<float, access::write> outTex [[texture(3)]],
    constant float &sigmaThreshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float4 sum = sumTex.read(gid);
    float count = countTex.read(gid).r;

    if (count < 1.0) {
        outTex.write(float4(0, 0, 0, 1), gid);
        return;
    }

    float3 mean = sum.rgb / count;

    // Gentle contrast stretch for star visibility
    float contrast = 1.12;
    float3 result = (mean - 0.5) * contrast + 0.5;
    result = clamp(result, 0.0, 1.0);

    // Lift shadows slightly
    float meanLum = (result.r + result.g + result.b) / 3.0;
    result = clamp(result + 0.02 * (1.0 - meanLum), 0.0, 1.0);

    outTex.write(float4(result, 1.0), gid);
}

// Parallel SSD: each thread computes per-pixel SSD; caller averages on CPU
kernel void computeSSD(
    texture2d<float, access::read> refTex [[texture(0)]],
    texture2d<float, access::read> candTex [[texture(1)]],
    texture2d<float, access::write> resultTex [[texture(2)]],
    constant float2 &shift [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint w = refTex.get_width();
    uint h = refTex.get_height();

    int sx = int(gid.x) - int(round(shift.x));
    int sy = int(gid.y) - int(round(shift.y));

    float localSSD = 0.0;
    if (sx >= 0 && sx < int(w) && sy >= 0 && sy < int(h)) {
        float3 r = refTex.read(gid).rgb;
        float3 c = candTex.read(uint2(sx, sy)).rgb;
        float3 d = r - c;
        localSSD = dot(d, d);
    }

    resultTex.write(localSSD, gid);
}

// High-pixel mode: clean average output (no astro contrast/shadow tweaks)
kernel void averageOutput(
    texture2d<float, access::read> sumTex [[texture(0)]],
    texture2d<float, access::read> sumSqTex [[texture(1)]],
    texture2d<float, access::read> countTex [[texture(2)]],
    texture2d<float, access::write> outTex [[texture(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float4 sum = sumTex.read(gid);
    float count = countTex.read(gid).r;
    if (count < 1.0) {
        outTex.write(float4(0, 0, 0, 1), gid);
        return;
    }
    float3 mean = sum.rgb / count;
    outTex.write(float4(mean, 1.0), gid);
}

// Unsharp mask sharpening for high-pixel mode
// out = original + amount * (original - blurred)
kernel void sharpenOutput(
    texture2d<float, access::read> inTex [[texture(0)]],
    texture2d<float, access::write> outTex [[texture(1)]],
    constant float &amount [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint w = inTex.get_width();
    uint h = inTex.get_height();
    uint x = gid.x, y = gid.y;

    float3 center = inTex.read(gid).rgb;

    // 3x3 box blur
    float3 blur = float3(0.0);
    float n = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int px = int(x) + dx;
            int py = int(y) + dy;
            if (px >= 0 && px < int(w) && py >= 0 && py < int(h)) {
                blur += inTex.read(uint2(px, py)).rgb;
                n += 1.0;
            }
        }
    }
    blur /= max(n, 1.0);

    float3 result = center + amount * (center - blur);
    result = clamp(result, 0.0, 1.0);
    outTex.write(float4(result, 1.0), gid);
}
