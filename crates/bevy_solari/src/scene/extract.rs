use super::RaytracingMesh3d;
use bevy_asset::{AssetId, Assets};
use bevy_camera::visibility::InheritedVisibility;
use bevy_derive::Deref;
use bevy_ecs::{
    entity::{Entity, EntityHashSet},
    query::With,
    resource::Resource,
    system::{Commands, Query},
};
use bevy_pbr::{MeshMaterial3d, StandardMaterial};
use bevy_platform::collections::HashMap;
use bevy_render::{extract_resource::ExtractResource, sync_world::RenderEntity, Extract};
use bevy_transform::components::GlobalTransform;

pub fn extract_raytracing_scene(
    instances: Extract<
        Query<(
            RenderEntity,
            &RaytracingMesh3d,
            &MeshMaterial3d<StandardMaterial>,
            &GlobalTransform,
            &InheritedVisibility,
        )>,
    >,
    existing: Query<Entity, With<RaytracingMesh3d>>,
    mut commands: Commands,
) {
    let mut extracted_this_frame = EntityHashSet::default();

    for (render_entity, mesh, material, transform, inherited_visibility) in &instances {
        // Use InheritedVisibility (not ViewVisibility): rays escape the
        // camera frustum (reflections, GI), so frustum culling is wrong here.
        if !inherited_visibility.get() {
            continue;
        }

        commands
            .entity(render_entity)
            .insert((mesh.clone(), material.clone(), *transform));
        extracted_this_frame.insert(render_entity);
    }

    // Sweep render-world entities for two cases SyncToRenderWorld
    // doesn't cover: RaytracingMesh3d removed without despawn, and
    // visibility flipping to hidden. Despawns are handled by sync.
    for render_entity in &existing {
        if !extracted_this_frame.contains(&render_entity) {
            commands
                .entity(render_entity)
                .try_remove::<RaytracingMesh3d>();
        }
    }
}

#[derive(Resource, Deref, Default)]
pub struct StandardMaterialAssets(HashMap<AssetId<StandardMaterial>, StandardMaterial>);

impl ExtractResource for StandardMaterialAssets {
    type Source = Assets<StandardMaterial>;

    fn extract_resource(source: &Self::Source) -> Self {
        Self(
            source
                .iter()
                .map(|(asset_id, material)| (asset_id, material.clone()))
                .collect(),
        )
    }
}
