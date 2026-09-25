//! Vector side of the cache: one Qdrant point per entry, holding the prompt
//! embedding and the exact-match `params` key.

use anyhow::{bail, Context, Result};
use qdrant_client::{
    qdrant::{
        point_id::PointIdOptions, vectors_config, Condition, CreateCollectionBuilder,
        CreateFieldIndexCollectionBuilder, DeletePointsBuilder, Distance, FieldType, Filter,
        PointId, PointStruct, PointsIdsList, QueryPointsBuilder, with_payload_selector::SelectorOptions,
        UpsertPointsBuilder, VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use serde_json::json;
use tracing::{info, instrument, Span};

pub struct VectorIndex {
    client: Qdrant,
    collection: String,
}

impl VectorIndex {
    /// Connects and makes sure the collection exists with the expected vector size.
    pub async fn connect(url: &str, collection: &str, dim: u64) -> Result<Self> {
        let client = Qdrant::from_url(url).build().context("building Qdrant client")?;

        if client.collection_exists(collection).await? {
            // Vectors from a different embedding model can't be compared, so
            // refuse to start rather than fail on every upsert.
            let existing = client
                .collection_info(collection)
                .await?
                .result
                .and_then(|r| r.config)
                .and_then(|c| c.params)
                .and_then(|p| p.vectors_config)
                .and_then(|v| v.config);
            if let Some(vectors_config::Config::Params(p)) = existing {
                if p.size != dim {
                    bail!(
                        "Qdrant collection {collection:?} holds {}-dim vectors but EMBEDDING_DIM is {dim}. \
                         Set QDRANT_COLLECTION to a new name, or delete the old collection.",
                        p.size
                    );
                }
            }
        } else {
            client
                .create_collection(
                    CreateCollectionBuilder::new(collection)
                        .vectors_config(VectorParamsBuilder::new(dim, Distance::Cosine)),
                )
                .await
                .context("creating Qdrant collection")?;
            // Every lookup filters on `params`, so index it.
            client
                .create_field_index(CreateFieldIndexCollectionBuilder::new(
                    collection,
                    "params",
                    FieldType::Keyword,
                ))
                .await
                .context("creating Qdrant payload index")?;
            info!(collection, dim, "created Qdrant collection");
        }

        Ok(Self { client, collection: collection.to_string() })
    }

    /// Up to `limit` entries with identical params and similarity >= `threshold`,
    /// most similar first, as `(id, similarity, prompt)`.
    #[instrument(name = "qdrant.query", skip_all, fields(otel.kind = "client", db.system = "qdrant", limit, threshold, candidates))]
    pub async fn candidates(
        &self,
        vector: Vec<f32>,
        params: &str,
        threshold: f32,
        limit: u64,
    ) -> Result<Vec<(String, f32, String)>> {
        let res = self
            .client
            .query(
                QueryPointsBuilder::new(&self.collection)
                    .query(vector)
                    .limit(limit)
                    .score_threshold(threshold)
                    .filter(Filter::must([Condition::matches("params", params.to_string())]))
                    .with_payload(SelectorOptions::Include(vec!["prompt".to_string()].into())),
            )
            .await
            .context("querying Qdrant")?;
        Span::current().record("candidates", res.result.len());

        res.result
            .into_iter()
            .map(|point| {
                let id = point_id_string(point.id).context("Qdrant returned a point without a UUID id")?;
                let prompt = point
                    .payload
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .cloned()
                    .unwrap_or_default();
                Ok((id, point.score, prompt))
            })
            .collect()
    }

    #[instrument(name = "qdrant.upsert", skip_all, fields(otel.kind = "client", db.system = "qdrant"))]
    pub async fn insert(&self, id: &str, vector: Vec<f32>, prompt: &str, params: &str) -> Result<()> {
        let payload = Payload::try_from(json!({ "prompt": prompt, "params": params }))?;
        self.client
            .upsert_points(
                UpsertPointsBuilder::new(&self.collection, vec![PointStruct::new(id.to_string(), vector, payload)])
                    .wait(true),
            )
            .await
            .context("upserting Qdrant point")?;
        Ok(())
    }

    #[instrument(name = "qdrant.delete", skip_all, fields(otel.kind = "client", db.system = "qdrant"))]
    pub async fn delete(&self, id: &str) -> Result<()> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection)
                    .points(PointsIdsList { ids: vec![id.to_string().into()] }),
            )
            .await
            .context("deleting Qdrant point")?;
        Ok(())
    }
}

fn point_id_string(id: Option<PointId>) -> Option<String> {
    match id?.point_id_options? {
        PointIdOptions::Uuid(uuid) => Some(uuid),
        PointIdOptions::Num(_) => None,
    }
}
