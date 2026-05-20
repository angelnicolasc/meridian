# Setting up the GPU CI runner

The `cuda.yml` workflow targets a self-hosted runner labelled `gpu` because
GitHub-hosted runners (as of June 2026) do not provide CUDA-capable GPUs in
the free tier. This page documents how to provision the runner.

## Hardware requirements

| Component | Minimum             | Recommended      |
|-----------|---------------------|------------------|
| GPU       | NVIDIA L4 / A10     | H100 / B200      |
| Driver    | 555.x               | 555.x or newer   |
| CUDA      | 12.6                | 12.6             |
| Disk      | 80 GiB SSD          | 200 GiB NVMe     |
| RAM       | 32 GiB              | 64 GiB           |

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

`./run.sh` once interactively; trigger the `cuda.yml` workflow from a PR
and confirm `nvidia-smi` prints the expected device.

## Security note

Self-hosted runners are documented as a footgun on public repositories
because any PR author can run arbitrary code on them. Meridian mitigates
this with the workflow gate
`if: github.repository_owner == 'angelnicolasc'` — PRs from forks do not
trigger the CUDA job. Maintainers may opt-in per-PR with a `safe-to-test`
label gate if community contribution volume grows.

See: <https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners#self-hosted-runner-security>.
