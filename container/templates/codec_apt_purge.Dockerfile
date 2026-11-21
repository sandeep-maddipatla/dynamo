{#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Purge the distro ffmpeg + media-codec packages by an explicit name match, shared
# by vllm_runtime.Dockerfile and sglang_runtime.Dockerfile. Set
# codec_purge_dependents first: the exact packages apt also removes because they
# depend on a matched one. A simulated purge runs first, and the build fails
# unless it removes every matched package and nothing outside the match and
# codec_purge_dependents, so a new dependent in a later base stops the build
# instead of leaving the image unnoticed. Deliberately no autoremove: it sweeps
# auto-installed toolchains (gcc/make, which torch.inductor/Triton JIT need).
-#}
RUN set -eux; \
    purge=$(dpkg-query -W -f='${Package}\n' 2>/dev/null \
        | grep -E '^(ffmpeg|libav[a-z]|libsw[a-z]|libpostproc|libx264|libx265|libmp3lame|libaom|libdav1d|libvpx|libtheora|libvorbis|libopus|libsoxr|libcaca|libcdio|libzvbi|libgme|libvidstab|libdc1394|libraw1394|libiec61883|libtwolame|libshine|libsrt[0-9]|libudfread|libsvtav1|libbs2b|librubberband|libchromaprint|libcodec2|libgsm|libass[0-9]|libbluray|libxvidcore|libflite|librist[0-9]|libmbedcrypto|libmbedtls|libmbedx509|libcjson)' \
        || true); \
    dependents="{{ codec_purge_dependents }}"; \
    if [ -n "$purge" ]; then \
        removed=$(apt-get -s purge $purge | awk '/^(Remv|Purg) /{sub(/:.*/, "", $2); print $2}'); \
        [ -n "$removed" ] || { echo "ERROR: a simulated purge of the matched packages removes nothing" >&2; exit 1; }; \
        kept=$(printf '%s\n' $purge | grep -vxF "$removed" || true); \
        extra=$(printf '%s\n' $removed | grep -vxF "$(printf '%s\n' $purge $dependents)" || true); \
        [ -z "$kept$extra" ] || { echo "ERROR: the purge would keep: ${kept:-none}; and also remove: ${extra:-none}" >&2; exit 1; }; \
        DEBIAN_FRONTEND=noninteractive apt-get purge -y $purge; \
    fi; \
    rm -rf /var/lib/apt/lists/*
