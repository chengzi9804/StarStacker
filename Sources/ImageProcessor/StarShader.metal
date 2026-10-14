#include <metal_stdlib>
using namespace metal;

// Accumulate one shifted frame into running sum, sum-of-squares, and count.
// sumTex / sumSqTex / countTex are rgba32Float; only .r of sumSq/count is used.
kernel void accumulateWithShift(
    texture2d<float, access::write> sumTex [[texture(0)]],
    texture2d<float, access::write> sumSqTex [[texture(1)]],
    texture2d<float, access::write> countTex [[texture(2)]],
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

// Sigma-clip average output: mean + stddev, then gentle contrast for star visibility.
// (Full iterative sigma-clip would require a second pass; this first pass computes
//  the statistics, and a production app would loop clip-recompute.)
kernel void sigmaClipOutput(
    texture2d<float, access::read> sumTex [[texture(0)]],
    texture2d<float, access::read> sumSqTex [[texture(1)]],
    texture2d<float, access::read> countTex [[texture(2)]],
    texture2d<float, access::write> outTex [[texture(3)]],
    constant float &sigmaThreshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float4 sum = sumTex.read(gid);
    float sumSq = sumSqTex.read(gid).r;
    float count = countTex.read(gid).r;

    if (count < 1.0) {
        outTex.write(float4(0, 0, 0, 1), gid);
        return;
    }

    float3 mean = sum.rgb / count;
    float meanLum = (mean.r + mean.g + mean.b) / 3.0;
    float var = sumSq / count - meanLum * meanLum;
    var = max(var, 0.0);
    float stddev = sqrt(var);

    // Gentle contrast stretch centered on mean
    float contrast = 1.12;
    float3 result = (mean - 0.5) * contrast + 0.5;
    result = clamp(result, 0.0, 1.0);

    // Boost dark areas slightly (star visibility)
    float shadowLift = 0.02;
    result = clamp(result + shadowLift * (1.0 - meanLum), 0.0, 1.0);

    outTex.write(float4(result, 1.0), gid);
}

// Compute SSD between reference and shifted candidate.
// Output: 1x1 texture with SSD in .r channel.
kernel void computeSSD(
    texture2d<float, access::read> refTex [[texture(0)]],
    texture2d<float, access::read> candTex [[texture(1)]],
    texture2d<float, access::write> resultTex [[texture(2)]],
    constant float2 &shift [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint w = refTex.get_width();
    uint h = refTex.get_height();

    // Sample center 50% region on a grid for speed
    int minX = int(w) / 4;
    int maxX = int(w) * 3 / 4;
    int minY = int(h) / 4;
    int maxY = int(h) * 3 / 4;
    int step = 8;

    float ssd = 0.0;
    int samples = 0;

    for (int y = minY; y < maxY; y += step) {
        for (int x = minX; x < maxX; x += step) {
            int sx = x - int(round(shift.x));
            int sy = y - int(round(shift.y));
            if (sx < 0 || sx >= int(w) || sy < 0 || sy >= int(h)) continue;

            float3 r = refTex.read(uint2(x, y)).rgb;
            float3 c = candTex.read(uint2(sx, sy)).rgb;
            float3 d = r - c;
            ssd += dot(d, d);
            samples += 1;
        }
    }

    if (samples > 0) ssd /= float(samples);
    resultTex.write(float4(ssd, 0, 0, 1), uint2(0, 0));
}
