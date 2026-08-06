# helm/

Helm chart packaging (mostly) of the same resources as `infra/k8s/base/`, for
teams that prefer `helm install` over `kubectl apply -k`. Pick one, don't run
both against the same cluster/namespace.

```bash
helm lint infra/helm/dev-repo-template
helm install dev-repo-template infra/helm/dev-repo-template -f infra/helm/dev-repo-template/values.yaml
```
