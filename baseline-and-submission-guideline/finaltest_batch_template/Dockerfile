# ATM26 Final Test Phase: batch Docker packing template.
#
# Dual-mode: on the real Grand-Challenge platform the container receives ONE
# case and behaves like a standard per-case algorithm; on the ATM26 local
# evaluation platform (org.atm26.batch=1) it receives the whole test set and
# performs internal sequential inference - the model is loaded ONCE and every
# case is processed in a loop with explicit GPU-VRAM/memory hygiene.
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

# Required by the ATM26 local platform: selects the batch execution path.
LABEL org.atm26.batch="1"
# Standard Grand Challenge exec interface.
LABEL org.grand-challenge.api-method="exec"

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
# Torch FIRST from the cu118 index: installing nnunetv2's requirements
# through a generic PyPI mirror would resolve torch to the default cu124
# wheels and stack a second torch on top of the base's (~11GB of waste).
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    torch==2.5.1+cu118 torchvision==0.20.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "/opt/app/inference.py"]
