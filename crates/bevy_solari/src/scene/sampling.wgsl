#define_import_path bevy_solari::sampling

#import bevy_pbr::lighting::D_GGX
#import bevy_pbr::utils::{rand_f, rand_vec2f, rand_u, rand_range_u}
#import bevy_render::maths::{PI_2, orthonormalize}
#import bevy_solari::scene_bindings::{trace_ray, RAY_T_MIN, RAY_T_MAX, RAY_NO_CULL, tlas, materials, material_ids, light_sources, directional_lights, spot_lights, LightSource, LIGHT_SOURCE_KIND_MASK, LIGHT_SOURCE_KIND_EMISSIVE_MESH, LIGHT_SOURCE_KIND_DIRECTIONAL, LIGHT_SOURCE_KIND_SPOT, ALPHA_MODE_OPAQUE, ALPHA_MODE_MASK, resolve_triangle_data_full, ResolvedRayHitFull}

fn power_heuristic(f: f32, g: f32) -> f32 {
    return balance_heuristic(f * f, g * g);
}

fn balance_heuristic(f: f32, g: f32) -> f32 {
    let sum = f + g;
    if sum == 0.0 {
        return 0.0;
    }
    return max(0.0, f / sum);
}

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 1)
fn sample_ggx_vndf(wi_tangent: vec3<f32>, roughness: f32, rng: ptr<function, u32>) -> vec3<f32> {
    if roughness <= 0.001 {
        return vec3(-wi_tangent.xy, wi_tangent.z);
    }

    let i = wi_tangent;
    let rand = rand_vec2f(rng);
    let i_std = normalize(vec3(i.xy * roughness, i.z));
    let phi = PI_2 * rand.x;
    let a = roughness;
    let s = 1.0 + length(vec2(i.xy));
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
    let b = select(i_std.z, k * i_std.z, i.z > 0.0);
    let z = fma(1.0 - rand.y, 1.0 + b, -b);
    let sin_theta = sqrt(saturate(1.0 - z * z));
    let o_std = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    let m_std = i_std + o_std;
    let m = normalize(vec3(m_std.xy * roughness, m_std.z));
    return 2.0 * dot(i, m) * m - i;
}

// https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf (Listing 2)
fn ggx_vndf_pdf(wi_tangent: vec3<f32>, wo_tangent: vec3<f32>, roughness: f32) -> f32 {
    let i = wi_tangent;
    let o = wo_tangent;
    let m = normalize(i + o);
    let ndf = D_GGX(roughness, saturate(m.z));
    let ai = roughness * i.xy;
    let len2 = dot(ai, ai);
    let t = sqrt(len2 + i.z * i.z);
    if i.z >= 0.0 {
        let a = roughness;
        let s = 1.0 + length(i.xy);
        let a2 = a * a;
        let s2 = s * s;
        let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
        return ndf / (2.0 * (k * i.z + t));
    }
    return ndf * (t - i.z) / (2.0 * len2);
}

struct LightSample {
    light_id: u32,
    seed: u32,
}

struct ResolvedLightSample {
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    radiance: vec3<f32>,
    inverse_pdf: f32,
    // Spot-light cone parameters. For non-spot lights, `cone_axis` is
    // zero, the cosines are sentinel values that make the cone
    // smoothstep evaluate to 1, and `range_squared` is set to a huge
    // value so the cutoff never triggers.
    cone_axis: vec3<f32>,
    inner_cos: f32,
    outer_cos: f32,
    range_squared: f32,
}

struct LightContribution {
    radiance: vec3<f32>,
    inverse_pdf: f32,
    wi: vec3<f32>,
    brdf_rays_can_hit: bool,
}

struct LightContributionNoPdf {
    radiance: vec3<f32>,
    wi: vec3<f32>,
}

struct GenerateRandomLightSampleResult {
    light_sample: LightSample,
    resolved_light_sample: ResolvedLightSample,
}

fn sample_random_light(ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, rng: ptr<function, u32>) -> LightContribution {
    let sample = generate_random_light_sample(rng);
    var light_contribution = calculate_resolved_light_contribution(sample.resolved_light_sample, ray_origin, origin_world_normal);
    light_contribution.radiance *= trace_light_visibility(ray_origin, sample.resolved_light_sample.world_position);
    return light_contribution;
}

fn random_emissive_light_pdf(hit: ResolvedRayHitFull) -> f32 {
    let light_count = arrayLength(&light_sources);
    return 1.0 / (f32(light_count) * f32(hit.triangle_count) * hit.triangle_area);
}

fn generate_random_light_sample(rng: ptr<function, u32>) -> GenerateRandomLightSampleResult {
    let light_count = arrayLength(&light_sources);
    let light_id = rand_range_u(light_count, rng);

    let light_source = light_sources[light_id];

    var triangle_id = 0u;
    if (light_source.kind & LIGHT_SOURCE_KIND_MASK) == LIGHT_SOURCE_KIND_EMISSIVE_MESH {
        let triangle_count = light_source.kind >> 2u;
        triangle_id = rand_range_u(triangle_count, rng);
    }

    let seed = rand_u(rng);
    let light_sample = LightSample((light_id << 16u) | triangle_id, seed);

    var resolved_light_sample = resolve_light_sample(light_sample, light_source);
    resolved_light_sample.inverse_pdf *= f32(light_count);

    return GenerateRandomLightSampleResult(light_sample, resolved_light_sample);
}

fn resolve_light_sample(light_sample: LightSample, light_source: LightSource) -> ResolvedLightSample {
    let kind = light_source.kind & LIGHT_SOURCE_KIND_MASK;
    if kind == LIGHT_SOURCE_KIND_DIRECTIONAL {
        let directional_light = directional_lights[light_source.id];

#ifndef NO_DIRECTIONAL_LIGHT_SOFT_SHADOWS
        // Sample a random direction within a cone whose base is the sun approximated as a disk
        // https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec30%3A305
        var rng = light_sample.seed;
        let random = rand_vec2f(&rng);
        let cos_theta = (1.0 - random.x) + random.x * directional_light.cos_theta_max;
        let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        let phi = random.y * PI_2;
        let x = cos(phi) * sin_theta;
        let y = sin(phi) * sin_theta;
        var direction_to_light = vec3(x, y, cos_theta);

        // Rotate the ray so that the cone it was sampled from is aligned with the light direction
        direction_to_light = orthonormalize(directional_light.direction_to_light) * direction_to_light;
#else
        let direction_to_light = directional_light.direction_to_light;
#endif

        return ResolvedLightSample(
            vec4(direction_to_light, 0.0),
            -direction_to_light,
            directional_light.luminance,
            directional_light.inverse_pdf,
            // Cone + range unused for directional lights: sentinels make
            // the smoothstep return 1 and the range cutoff never trip.
            vec3(0.0),
            -2.0,
            -2.0,
            3.402823e38,
        );
    } else if kind == LIGHT_SOURCE_KIND_SPOT {
        // Spot light is a delta point with a cone-shaped intensity
        // profile. We return the light's world position with w = 1, so
        // calculate_resolved_light_contribution applies the geometric
        // 1/dist² as for any point light. The world_normal is set to
        // an arbitrary unit vector — `cos_theta_light` is overridden
        // to 1 inside calculate_resolved_light_contribution when cone
        // params are non-trivial. Cone attenuation is computed there.
        let spot_light = spot_lights[light_source.id];
        return ResolvedLightSample(
            vec4(spot_light.position, 1.0),
            spot_light.direction,
            spot_light.luminance,
            1.0,
            spot_light.direction,
            spot_light.inner_cos,
            spot_light.outer_cos,
            spot_light.range_squared,
        );
    } else {
        let triangle_count = light_source.kind >> 2u;
        let triangle_id = light_sample.light_id & 0xFFFFu;
        let barycentrics = triangle_barycentrics(light_sample.seed);
        let triangle_data = resolve_triangle_data_full(light_source.id, triangle_id, barycentrics);

        return ResolvedLightSample(
            vec4(triangle_data.world_position, 1.0),
            triangle_data.world_normal,
            triangle_data.material.emissive.rgb,
            f32(triangle_count) * triangle_data.triangle_area,
            vec3(0.0),
            -2.0,
            -2.0,
            3.402823e38,
        );
    }
}

fn calculate_resolved_light_contribution(resolved_light_sample: ResolvedLightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContribution {
    let ray = resolved_light_sample.world_position.xyz - (resolved_light_sample.world_position.w * ray_origin);
    let light_distance = length(ray);
    let wi = ray / light_distance;

    let cos_theta_origin = saturate(dot(wi, origin_world_normal));

    // Spot lights are delta points: no surface normal, so cos_theta_light
    // is 1, and a separate cone falloff replaces the surface attenuation.
    // Detect via `inner_cos > outer_cos` — non-spot samples store
    // (-2, -2) as sentinels.
    let is_spot = resolved_light_sample.inner_cos > resolved_light_sample.outer_cos;
    var cos_theta_light: f32;
    var cone_factor: f32;
    if is_spot {
        cos_theta_light = 1.0;
        let cos_axis = dot(resolved_light_sample.cone_axis, -wi);
        cone_factor = smoothstep(
            resolved_light_sample.outer_cos,
            resolved_light_sample.inner_cos,
            cos_axis,
        );
    } else {
        cos_theta_light = saturate(dot(-wi, resolved_light_sample.world_normal));
        cone_factor = 1.0;
    }

    let light_distance_squared = light_distance * light_distance;

    // Hard cutoff beyond the light's range — for directional/emissive
    // this never trips because range_squared is set to f32::MAX.
    let range_factor = select(0.0, 1.0, light_distance_squared <= resolved_light_sample.range_squared);

    let radiance = resolved_light_sample.radiance * cos_theta_origin * cone_factor * range_factor * (cos_theta_light / light_distance_squared);

    return LightContribution(radiance, resolved_light_sample.inverse_pdf, wi, resolved_light_sample.world_position.w == 1.0);
}

fn resolve_and_calculate_light_contribution(light_sample: LightSample, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightContributionNoPdf {
    let resolved_light_sample = resolve_light_sample(light_sample, light_sources[light_sample.light_id >> 16u]);
    let light_contribution = calculate_resolved_light_contribution(resolved_light_sample, ray_origin, origin_world_normal);
    return LightContributionNoPdf(light_contribution.radiance, light_contribution.wi);
}

fn trace_light_visibility(ray_origin: vec3<f32>, light_sample_world_position: vec4<f32>) -> f32 {
    var ray_direction = light_sample_world_position.xyz;
    var ray_t_max = RAY_T_MAX;

    if light_sample_world_position.w == 1.0 {
        let ray = ray_direction - ray_origin;
        let dist = length(ray);
        ray_direction = ray / dist;
        ray_t_max = dist - RAY_T_MIN - RAY_T_MIN;
    }

    if ray_t_max < RAY_T_MIN { return 0.0; }

    // Manual candidate loop so we can skip alpha-masked / blend
    // surfaces. Without RAY_FLAG_OPAQUE, every triangle hit becomes a
    // candidate; we explicitly confirm only those that should occlude.
    let ray = RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, RAY_NO_CULL, RAY_T_MIN, ray_t_max, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    while rayQueryProceed(&rq) {
        let candidate = rayQueryGetCandidateIntersection(&rq);
        let material_id = material_ids[candidate.instance_index];
        let material = materials[material_id];
        if material.alpha_mode == ALPHA_MODE_OPAQUE {
            rayQueryConfirmIntersection(&rq);
        } else if material.alpha_mode == ALPHA_MODE_MASK
            && material.base_color_alpha >= material.alpha_cutoff {
            rayQueryConfirmIntersection(&rq);
        }
        // Mask below cutoff or Blend: rays pass through.
    }
    let hit = rayQueryGetCommittedIntersection(&rq);
    return f32(hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

fn trace_point_visibility(ray_origin: vec3<f32>, point: vec3<f32>) -> f32 {
    let ray = point - ray_origin;
    let dist = length(ray);
    let ray_direction = ray / dist;

    let ray_t_max = dist - RAY_T_MIN - RAY_T_MIN;
    if ray_t_max < RAY_T_MIN { return 0.0; }

    let ray_hit = trace_ray(ray_origin, ray_direction, RAY_T_MIN, ray_t_max, RAY_FLAG_TERMINATE_ON_FIRST_HIT);
    return f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
}

// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec22%3A297
fn triangle_barycentrics(seed: u32) -> vec3<f32> {
    var rng = seed;
    var barycentrics = rand_vec2f(&rng);
    if barycentrics.x + barycentrics.y > 1.0 { barycentrics = 1.0 - barycentrics; }
    return vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
}
