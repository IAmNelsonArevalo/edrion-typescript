# Edrion Nest Lab

Executable NestJS consumer used by Edrion's integration QA.

It validates that `@edrion/nestjs` works outside its own package boundary, including dependency injection, module lifecycle, runtime initialization, health reporting and shutdown.

```bash
pnpm --filter @edrion/nest-lab test
pnpm --filter @edrion/nest-lab build
```
