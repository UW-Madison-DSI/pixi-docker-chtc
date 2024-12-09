FROM ghcr.io/prefix-dev/pixi:jammy AS build

WORKDIR /app
COPY . .
RUN pixi install --locked --environment prod
RUN echo "#!/bin/bash" > /app/entrypoint.sh && \
    pixi shell-hook --environment prod -s bash >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh

FROM ghcr.io/prefix-dev/pixi:jammy-cuda-11.8.0 AS production

WORKDIR /app
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY --from=build /app/pixi.toml /app/pixi.toml
COPY --from=build /app/pixi.lock /app/pixi.lock
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
COPY ./src /app/src

EXPOSE 8000
ENTRYPOINT [ "/app/entrypoint.sh" ]
