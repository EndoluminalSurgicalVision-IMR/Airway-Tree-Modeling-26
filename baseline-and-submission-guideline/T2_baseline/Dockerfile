# ATM26 Track 2 single-fold nnU-Net leaderboard baseline.
FROM atm26-nnunetv2:2.6.4-cuda12.4

ENV PYTHONUNBUFFERED=1
ENV PATH="/home/user/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

USER root
RUN rm -rf /opt/app
WORKDIR /opt/app

COPY --chown=user:user inference.py /opt/app/inference.py
COPY --chown=user:user resources /opt/app/resources

RUN test -f "/opt/app/resources/nnUNet_ckpts/fold_0/checkpoint_best.pth" && \
    for fold in 1 2 3 4; do \
      test ! -e "/opt/app/resources/nnUNet_ckpts/fold_${fold}"; \
    done

USER user
WORKDIR /opt/app

ENTRYPOINT ["python", "/opt/app/inference.py"]
