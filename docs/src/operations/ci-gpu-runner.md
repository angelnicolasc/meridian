# GPU CI runner setup

The `cuda.yml` workflow targets a self-hosted runner labelled `gpu` because
GitHub-hosted runners do not provide CUDA-capable GPUs in the free tier.
This page documents how to provision and secure the runner.

## Hardware requirements

| Component | Minimum             | Recommended      |
|-----------|---------------------|------------------|
| GPU       | NVIDIA L4 / A10     | H100 / B200      |
| Driver    | 555.x               | 555.x or newer   |
| CUDA      | 12.6                | 12.6             |
| Disk      | 80 GiB SSD          | 200 GiB NVMe     |
| RAM       | 32 GiB              | 64 GiB           |

## Required secrets

| Secret | Where set | Purpose |
|--------|-----------|---------|
| *(none required for GPU runner itself)* | — | The runner authenticates via GitHub App token |
| `RELEASE_PLZ_TOKEN` | Repository secrets | Allows release-plz to push tags |
| `CARGO_REGISTRY_TOKEN` | Repository secrets | crates.io publish (future) |

The runner registration token is generated once from the GitHub Actions UI and
is not stored as a persistent secret — it expires after one hour and is only
used during `./config.sh`.

## Provisioning

```bash
# On the Linux host with the GPU
curl -O https://github.com/actions/runner/releases/download/v2.319.0/actions-runner-linux-x64-2.319.0.tar.gz
mkdir actions-runner && cd actions-runner
tar xzf ../actions-runner-linux-x64-2.319.0.tar.gz

./config.sh \
    --url https://github.com/angelnicolasc/meridian \
    --token <REGISTRATION_TOKEN> \
    --labels self-hosted,linux,x64,gpu \
    --unattended

sudo ./svc.sh install
sudo ./svc.sh start
```

## Verification

Run `./run.sh` once interactively. Then trigger the `cuda.yml` workflow from
a branch and confirm `nvidia-smi` prints the expected device in the job logs.

## Blast radius and fork safety

Self-hosted runners execute arbitrary code from the workflow YAML. Meridian
mitigates this with a hard gate on every GPU job:

```yaml
if: github.repository_owner == 'angelnicolasc'
```

PRs from forks never trigger the CUDA workflow. Only pushes and PRs from the
`angelnicolasc` org are eligible.

**Who can trigger**: repository owners and collaborators with write access.  
**How to rotate the runner**: generate a new registration token from the
GitHub Actions UI, run `./config.sh --replace`, restart the service.

See the GitHub documentation on
[self-hosted runner security](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners#self-hosted-runner-security)
for a full threat model.
