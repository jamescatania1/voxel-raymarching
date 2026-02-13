@group(0) @binding(0) var tex_color: texture_2d<f32>;
@group(0) @binding(1) var main_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.uv = vec2(position.x + 1.0, 1.0 - position.y) * 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // let color = fxaa(in.position, in.uv);
    let color = textureSample(tex_color, main_sampler, in.uv).rgb;
    return vec4(color, 1.0);
}

const EDGE_THRESHOLD_MIN: f32 = 0.0312;
const EDGE_THRESHOLD_MAX: f32 = 0.125;
const EDGE_STEPS: array<f32, 10> = array<f32, 10>(1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0);
const SUB_PIXEL_BLENDING: f32 = 1.0;

fn fxaa(position: vec4<f32>, uv: vec2<f32>) -> vec3<f32> {
    let texel_size = 1.0 / vec2<f32>(textureDimensions(tex_color).xy);
    let dx = texel_size.x;
    let dy = texel_size.y;

    let color_ctr = textureSample(tex_color, main_sampler, uv).rgb;
    let luma_ctr = luma(color_ctr);

    let luma_n = tex_luma(uv, vec2(0., dy));
    let luma_e = tex_luma(uv, vec2(dx, 0.));
    let luma_s = tex_luma(uv, vec2(0., -dy));
    let luma_w = tex_luma(uv, vec2(-dx, 0.));

    let luma_min = min(min(min(min(luma_ctr, luma_n), luma_e), luma_s), luma_w);
    let luma_max = max(max(max(max(luma_ctr, luma_n), luma_e), luma_s), luma_w);

    let luma_range = luma_max - luma_min;

    if luma_range < max(EDGE_THRESHOLD_MIN, luma_max * EDGE_THRESHOLD_MAX) {
        return color_ctr;
    }

    let luma_ne = tex_luma(uv, vec2(dx, dy));
    let luma_se = tex_luma(uv, vec2(dx, -dy));
    let luma_nw = tex_luma(uv, vec2(-dx, dy));
    let luma_sw = tex_luma(uv, vec2(-dx, -dy));

    let horizontal = 2.0 * abs(luma_n + luma_s - 2.0 * luma_ctr)
        + abs(luma_ne + luma_se - 2.0 * luma_e)
        + abs(luma_nw + luma_sw - 2.0 * luma_w);
    let vertical = 2.0 * abs(luma_e + luma_w - 2.0 * luma_ctr)
        + abs(luma_ne + luma_nw - 2.0 * luma_n)
        + abs(luma_se + luma_sw - 2.0 * luma_s);

    let is_horizontal = horizontal >= vertical;
    let p_luma = select(luma_e, luma_n, is_horizontal);
    let n_luma = select(luma_w, luma_s, is_horizontal);
    let p_grad = abs(p_luma - luma_ctr);
    let n_grad = abs(n_luma - luma_ctr);

    var pixel_step = select(texel_size.x, texel_size.y, is_horizontal);

    pixel_step *= select(1.0, -1.0, p_grad < n_grad);
    let opposite_luma = select(p_luma, n_luma, p_grad < n_grad);
    let gradient = select(p_grad, n_grad, p_grad < n_grad);

    let edge_step = select(vec2(0., texel_size.y), vec2(texel_size.x, 0.), is_horizontal);
    let uv_edge = 0.5 * select(vec2(pixel_step, 0.), vec2(0., pixel_step), is_horizontal);

    let edge_luma = (luma_ctr + opposite_luma) * 0.5;
    let grad_threshold = gradient * 0.25;

    var p_uv = uv_edge + edge_step * EDGE_STEPS[0];
    var p_luma_delta = tex_luma(uv, p_uv) - edge_luma;
    var reached_end = abs(p_luma_delta) >= grad_threshold;

    for (var i = 1; i < 10; i++) {
        p_uv += edge_step * EDGE_STEPS[i];
        p_luma_delta = tex_luma(uv, p_uv) - edge_luma;
        reached_end = abs(p_luma_delta) >= grad_threshold;
    }
    if (!reached_end) {
        p_uv += edge_step * 2.0 * EDGE_STEPS[9];
    }

    var n_uv = uv_edge - edge_step * EDGE_STEPS[0];
    var n_luma_delta = tex_luma(uv, n_uv) - edge_luma;
    reached_end = abs(n_luma_delta) >= grad_threshold;

    for (var i = 1; i < 10; i++) {
        n_uv -= edge_step * EDGE_STEPS[i];
        n_luma_delta = tex_luma(uv, n_uv) - edge_luma;
        reached_end = abs(n_luma_delta) >= grad_threshold;
    }
    if (!reached_end) {
        n_uv -= edge_step * 2.0 * EDGE_STEPS[9];
    }

    let pos_distance = abs(select(p_uv.y - uv.y, p_uv.x - uv.x, is_horizontal));
    let neg_distance = abs(select(n_uv.y - uv.y, n_uv.x - uv.x, is_horizontal));

    let min_distance = min(pos_distance, neg_distance);
    let delta_sign = select(n_luma_delta >= 0., p_luma_delta >= 0., pos_distance <= neg_distance);

    var blend_factor = 0.0;
    if (delta_sign != (luma_ctr - edge_luma >= 0.)) {
        blend_factor = 0.5 - min_distance / (pos_distance + neg_distance);
    }

    // subpixel filter
    var subp_blend_factor = 2.0 * (luma_n + luma_e + luma_s + luma_w) + luma_ne + luma_nw + luma_se + luma_sw;
    subp_blend_factor = saturate(abs(subp_blend_factor / 12.0 - luma_ctr) / luma_range);
    subp_blend_factor = smoothstep(0.0, 1.0, subp_blend_factor);
    subp_blend_factor = subp_blend_factor * subp_blend_factor * SUB_PIXEL_BLENDING;

    blend_factor = max(blend_factor, subp_blend_factor);

    let final_offset = blend_factor * select(vec2(pixel_step, 0.), vec2(0., pixel_step), is_horizontal);

    let color = textureSample(tex_color, main_sampler, uv + final_offset).rgb;

    return color;
}

fn tex_luma(uv: vec2<f32>, offset: vec2<f32>) -> f32 {
    return luma(textureSample(tex_color, main_sampler, uv + offset).rgb);
}

const LUMA: vec3<f32> = vec3<f32>(0.299, 0.587, 0.114);

fn luma(v: vec3<f32>) -> f32 {
    return sqrt(dot(v, LUMA));
}
