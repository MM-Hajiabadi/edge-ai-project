# Dockerfile (Optional)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System requirements
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv \
    g++ make autoconf flex bison help2man libfl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build verilator 5.036 from source 
RUN git clone https://github.com/verilator/verilator.git /tmp/verilator \
    && cd /tmp/verilator \
    && git checkout v5.036 \
    && autoconf && ./configure && make -j2 && make install \
    && rm -rf /tmp/verilator

# Python dependencies
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip \
    && pip install cocotb pytest numpy scikit-learn pyyaml torch torchvision 
    
WORKDIR /workspace
COPY . .

CMD ["make"]
