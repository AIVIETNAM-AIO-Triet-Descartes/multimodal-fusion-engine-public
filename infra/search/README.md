# Local production-store stack

This stack pins Milvus `2.6.20` and Elasticsearch `9.4.4`. The Python clients are
PyMilvus `2.6.16` and elasticsearch-py `9.4.1`.

Budget roughly 6–8 GB of available RAM and several GB of disk for images plus local test
indexes. Inspect Docker Desktop resources before downloading. This compose file is for a
single-machine development environment: Elasticsearch security is disabled here, never in a
shared or production deployment.

The default text analyzer is deliberately named `aic_vi_dev_fallback_v1`. It uses the
standard tokenizer, lowercase, and ASCII folding. It is useful for development, but it is
not presented as a compatible Vietnamese linguistic plugin. Set
`MFE_ELASTICSEARCH_ANALYZER_MODE=plugin_required` only after the team installs and verifies
a plugin compatible with the pinned Elasticsearch server.

```powershell
Copy-Item infra/search/.env.example infra/search/.env
docker compose --env-file infra/search/.env -f infra/search/docker-compose.yml up -d
docker compose --env-file infra/search/.env -f infra/search/docker-compose.yml ps
python -m pip install -e ".[production-stores,dev]"
mfe-production-store ensure-schema
mfe-production-store health
```

Khi chưa có Docker nhưng muốn test vector local, cài `.[production-lite,dev]` và chạy
`scripts/ingest_btc_vectors.py`. Đây là Milvus Lite embedded cho local validation, không phải
standalone Milvus dùng chung.

To prove persistence, ingest a fixture, record `counts`, run `docker compose restart`, wait
for healthy services, then repeat the same queries and counts. Never use `down -v` unless
you explicitly intend to destroy the local indexes.

Safe stop and restart commands preserve named volumes:

```powershell
docker compose --env-file infra/search/.env -f infra/search/docker-compose.yml stop
docker compose --env-file infra/search/.env -f infra/search/docker-compose.yml restart
```
