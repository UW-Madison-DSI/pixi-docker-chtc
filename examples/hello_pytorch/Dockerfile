FROM ghcr.io/prefix-dev/pixi:noble AS build

WORKDIR /app
COPY . .
ENV CONDA_OVERRIDE_CUDA=12.0
RUN pixi install --locked --environment prod
RUN echo "#!/bin/bash" > /app/entrypoint.sh && \
    pixi shell-hook --environment prod -s bash >> /app/entrypoint.sh && \
    echo 'exec "$@"' >> /app/entrypoint.sh

FROM ghcr.io/prefix-dev/pixi:noble AS production

# ECHO REVERT
WORKDIR /app
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY --from=build /app/pixi.toml /app/pixi.toml
COPY --from=build /app/pixi.lock /app/pixi.lock
# Need the .gitignore if you want to use `pixi run` commands
COPY --from=build /app/.gitignore /app/.pixi/.gitignore
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
COPY ./src /app/src

EXPOSE 8000
ENTRYPOINT [ "/app/entrypoint.sh" ]
