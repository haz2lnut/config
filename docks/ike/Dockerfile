FROM alpine:latest
RUN apk update && apk upgrade && \
    apk add --no-cache \
    linux-headers busybox build-base \
    openssl-dev argp-standalone confuse-dev \
    gdb tcpdump \
    strongswan
