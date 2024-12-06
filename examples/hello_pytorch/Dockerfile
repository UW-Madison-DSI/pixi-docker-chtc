FROM ghcr.io/prefix-dev/pixi:jammy AS build

WORKDIR /app
COPY . .
RUN pixi install --locked -e prod
RUN pixi shell-hook -e prod -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh && \
    cat /shell-hook >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh

FROM ghcr.io/prefix-dev/pixi:jammy-cuda-11.8.0 as production

WORKDIR /app
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
COPY ./src /app/src

EXPOSE 8000
ENTRYPOINT [ "/app/entrypoint.sh" ]
