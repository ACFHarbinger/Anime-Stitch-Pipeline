# terraform/

Minimal Terraform skeleton for provisioning cloud resources this project
depends on (a container registry, a managed Postgres instance, a Kubernetes
cluster, etc.). No provider is wired up yet — this is a starting point, not
a real stack.

```bash
cd infra/terraform
terraform init
terraform plan -var-file=environments/dev.tfvars
terraform apply -var-file=environments/dev.tfvars
```

| File | Purpose |
| --- | --- |
| `versions.tf` | Required Terraform + provider versions, remote state backend (commented, fill in before first `init`) |
| `variables.tf` | Input variables |
| `main.tf` | Resources — currently empty, add your provider blocks and resources here |
| `outputs.tf` | Values to surface after `apply` (e.g. registry URL, cluster endpoint) |
| `environments/*.tfvars` | Per-environment variable values |

> **TODO:** Pick a cloud provider, uncomment/configure the matching provider
> block in `versions.tf`, and replace the placeholder resources in `main.tf`.
