FROM nvcr.io/nvidia/tritonserver:24.01-py3

WORKDIR /opt

RUN git clone https://github.com/sgl-project/sglang.git

WORKDIR /opt/sglang
RUN pip install --upgrade pip && \
    pip install -e "python[all]" && \
    pip install datasets

# The source tree is loaded into the serving process; keep it non-writable for
# group and others so a non-root user cannot replace installed code.
RUN chown -R root:root /opt/sglang && chmod -R go-w /opt/sglang
