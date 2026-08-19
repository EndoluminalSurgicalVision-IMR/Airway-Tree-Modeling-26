# ATM26 Final Test Phase: Track-1 leaderboard baseline, batch wrapper.
#
# Same single-fold nnU-Net baseline as T1_baseline, but declares the
# org.atm26.batch label: on the ATM26 local evaluation platform the
# container receives the WHOLE test set under /input and performs internal
# sequential inference (model loaded once, per-case loop with explicit
# GPU-VRAM/memory hygiene). On the standard Grand Challenge platform
# (one case per invocation) it still behaves like the per-case baseline.
# NOTE: the ATM26 evaluation platform's GPU driver supports CUDA up to
# 12.0 — the base image MUST be the CUDA 11.8 variant (cuda12.4 images
# will not run there).
FROM atm26-nnunetv2:2.6.4-cuda11.8

ENV PYTHONUNBUFFERED=1
ENV PATH="/home/user/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# ATM26 local platform: internal sequential (batch) execution.
LABEL org.atm26.batch="1"
# Standard Grand Challenge exec interface.
LABEL org.grand-challenge.api-method="exec"

USER root
RUN rm -rf /opt/app
WORKDIR /opt/app

COPY --chown=user:user inference.py /opt/app/inference.py
COPY --chown=user:user resources /opt/app/resources

USER user
WORKDIR /opt/app

ENTRYPOINT ["python", "/opt/app/inference.py"]
