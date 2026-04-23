This [Code Ocean](https://codeocean.com) Compute Capsule will allow you to run and reproduce the results of [Mask Gen](https://codeocean.allenneuraldynamics.org/capsule/8333757/tree) on your local machine<sup>1</sup>. Follow the instructions below, or consult [our knowledge base](https://docs.codeocean.com/user-guide/compute-capsule-basics/managing-capsules/exporting-capsules-to-your-local-machine) for more information. Don't hesitate to reach out to [Support](mailto:support@codeocean.com) if you have any questions.

<sup>1</sup> You may need access to additional hardware and/or software licenses.

# Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)
- [nvidia-container-runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu) for code that leverages the GPU

# Instructions

## Download attached Data Assets

In order to fetch the Data Asset(s) this Capsule depends on, download them into the Capsule's `data` folder:
* [mask-training-2025-05-23](https://codeocean.allenneuraldynamics.org/data-assets/051ddcaa-f7d6-4ab2-9130-a6b26e1ada68) should be downloaded to `data/mask-training-2025-05-23`
* [754615-mask-training-data](https://codeocean.allenneuraldynamics.org/data-assets/15b6b22c-8bbd-409e-951d-668e279da735) should be downloaded to `data/754615-mask-training-data`
* [mask-training-data-2025-05-13](https://codeocean.allenneuraldynamics.org/data-assets/2dd28ac4-5809-4863-b4ad-9d37143dd41a) should be downloaded to `data/mask-training-data-2025-05-13`
* [751474_mask-result-2025-06-02](https://codeocean.allenneuraldynamics.org/data-assets/3ba066c2-4c46-47f3-886d-410921ebcc6c) should be downloaded to `data/751474_mask-result-2025-06-02`
* [mask-training-data-2025-05-14](https://codeocean.allenneuraldynamics.org/data-assets/5129039d-ebba-49b9-a046-ba73eb4abbec) should be downloaded to `data/mask-training-data-2025-05-14`
* [mask-training-2025-05-31](https://codeocean.allenneuraldynamics.org/data-assets/a2a18c47-0d7c-4f73-a405-e7bf4455896e) should be downloaded to `data/mask-training-2025-05-31`
* [mask-training-2025-05-15](https://codeocean.allenneuraldynamics.org/data-assets/b0edd15e-1a5d-4408-8509-6da3dc38c497) should be downloaded to `data/mask-training-2025-05-15`
* [754615_mask-result-2025-06-02](https://codeocean.allenneuraldynamics.org/data-assets/b6aa13b7-656e-406e-9098-eb4dd6a91085) should be downloaded to `data/754615_mask-result-2025-06-02`
* [mask-result-2025-06-02](https://codeocean.allenneuraldynamics.org/data-assets/b77dbf95-e463-4d26-a45a-5a5cc4bafa57) should be downloaded to `data/mask-result-2025-06-02`
* [mask-training-2025-05-16](https://codeocean.allenneuraldynamics.org/data-assets/d7bb0857-0c00-481d-8fa6-bf6048f0a114) should be downloaded to `data/mask-training-2025-05-16`

## Log in to the Docker registry

In your terminal, execute the following command, providing your password or API key when prompted for it:
```shell
docker login -u cameron.arshadi@alleninstitute.org registry.codeocean.allenneuraldynamics.org
```

## Run the Capsule to reproduce the results

In your terminal, navigate to the folder where you've extracted the Capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm --gpus all \
  --workdir /code \
  --volume "$PWD/code":/code \
  --volume "$PWD/data":/data \
  --volume "$PWD/results":/results \
  --env AWS_ACCESS_KEY_ID=value \
  --env AWS_SECRET_ACCESS_KEY=value \
  --env AWS_DEFAULT_REGION=value \
  registry.codeocean.allenneuraldynamics.org/capsule/8732c547-1336-4261-a682-fd53c9f61786 \
  bash run '' '' '' ''
```

As secrets are required, replace all `value` occurances with your personal credentials prior to your run.
