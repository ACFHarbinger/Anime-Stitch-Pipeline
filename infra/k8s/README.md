# k8s/

Kustomize-based Kubernetes manifests: a `base/` layer plus per-environment
`overlays/`. Prefer this over hand-maintaining near-duplicate YAML per
environment.

```bash
kubectl apply -k infra/k8s/overlays/dev
kubectl apply -k infra/k8s/overlays/prod
```

| Directory | Purpose |
| --- | --- |
| `base/` | Environment-agnostic Deployment/Service/ConfigMap/Ingress |
| `overlays/dev/` | Dev patches: fewer replicas, `imagePullPolicy: Always`, dev host |
| `overlays/prod/` | Prod patches: replica count, resource limits, prod host |

> **TODO:** Point the `image:` field in `base/deployment.yaml` at your real
> container registry once one exists (see `infra/docker/`). The Helm chart
> in `infra/helm/` packages the same base manifests for teams that prefer
> `helm install` over `kubectl apply -k`.
