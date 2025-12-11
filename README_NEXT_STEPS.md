Next Steps

- Harden sandbox: For untrusted code, run node functions in containerized workers (Docker/Podman) with CPU/memory limits, seccomp, and minimal filesystem access.
- Prototype option: I can add a small Docker-based worker prototype that the engine can delegate to; it would accept serialized node code and state, run it inside an ephemeral container, and return results.
- Tests & CI: Add OOM/CPU stress tests and a CI job to validate worker isolation/termination.
- Docs: Document service-level tradeoffs and recommended production deployment steps in `README.md`.

