# DuxxDB Cloud — Solution Architecture & AWS Deployment

How to deploy the whole stack you've built (data plane, control plane, Studio,
document layer, frontend) on AWS — from a single EC2 box to a production
multi-AZ cluster — and exactly which AWS resources it needs.

---

## 1. What actually runs (component inventory)

| Process | Binary | Listens on | State | Role |
|---|---|---|---|---|
| **Data plane** | `duxx-server` | RESP `6379`, Studio `7072`, OTLP `4318`, metrics `9100`, replication `7075` | **Stateful** — workspaces on disk (`--tenants-dir`) | memory, retrieval, tool-cache, traces, evals, cost, **documents** |
| gRPC gateway | `duxx-grpc` | `50051` | stateless | gRPC surface (optional) |
| **Control plane** | `duxx-control` | HTTP `7070` (+ Console UI) | metadata in **Postgres** | orgs/projects/keys/members, **JWT issuance**, billing |
| **Frontend** | React SPA (static) | `80/443` | none | Console + Studio UIs |
| Object storage | S3 / MinIO | `9000` | **the file bytes** | document objects + snapshots |
| Metadata DB | PostgreSQL | `5432` | **control metadata** | `PostgresStore` behind the `Store` trait |
| Coordinator | etcd (optional) | `2379` | election state | leader election for HA |

**Two stateful stores, never mixed:** Postgres holds *control metadata*; the
data-plane node's disk holds *agent data* (workspaces). File **bytes** live in
S3; DuxxDB holds the *document intelligence* over them.

---

## 2. Production reference architecture (multi-AZ)

```
                              Route 53 (DNS)  ·  ACM (TLS certs)
                                        │
                    ┌───────────────────┴───────────────────┐
                    │            CloudFront (CDN)            │  ── frontend SPA
                    └───────────────────┬───────────────────┘     (S3 static site)
                                        │ HTTPS
                          ┌─────────────▼─────────────┐
                          │   ALB  (public subnets)   │  WAF, ACM, access logs→S3
                          │  /v1/* → control:7070     │
                          │  /studio/* → server:7072  │
                          │  /v1/traces → server:4318 │
                          └──────┬──────────────┬─────┘
            ══════════════════════│══════════════│════════════════ PRIVATE VPC
                       ┌──────────▼───┐   ┌──────▼───────────────┐
                       │ Control plane │   │  Data-plane nodes    │
                       │  ASG (EC2)    │   │  ASG / fixed (EC2)   │
                       │  duxx-control │   │  duxx-server         │
                       │  (stateless)  │   │  leader + followers  │
                       └──────┬────────┘   │  EBS gp3 (KMS) per   │
                              │            │  node = --tenants-dir│
                              │            └───┬──────────┬───────┘
                ┌─────────────▼──────┐   ┌─────▼────┐  ┌──▼─────────────┐
                │ RDS PostgreSQL     │   │   S3     │  │ etcd (optional)│
                │ Multi-AZ (KMS)     │   │ objects, │  │ leader election│
                │ control metadata   │   │ snapshots│  └────────────────┘
                └────────────────────┘   └──────────┘
        Secrets Manager: Ed25519 signing key · RDS creds · replication token
        KMS: encrypt EBS + RDS + S3 + Secrets    CloudWatch: logs + alarms
```

**Request flow:** browser → CloudFront (SPA) → ALB → control (`/v1/*`) or data
plane (`/studio/*`, `/v1/traces`). Agents connect to the data plane over RESP
(internal NLB or direct, JWT-authenticated). The control plane **signs** a
workspace JWT; the data plane **verifies** it (Ed25519 public key) and routes to
the tenant's isolated workspace.

---

## 3. AWS resources required (the checklist)

### Networking
- **VPC** (e.g. `10.0.0.0/16`) across **2–3 AZs**
- **Public subnets** (ALB, NAT) + **private subnets** (EC2, RDS)
- **Internet Gateway** + **NAT Gateway** (per AZ for HA)
- **Security Groups**: ALB-SG (443 from world), app-SG (7070/7072/4318 from ALB-SG; 6379/7075 from app-SG only), rds-SG (5432 from app-SG)
- (optional) **PrivateLink** endpoints for S3/Secrets/KMS — keep traffic off the internet

### Compute
- **EC2 Auto Scaling Group** — data-plane nodes (`duxx-server`), private subnets
- **EC2 (ASG)** — control-plane (`duxx-control`), private subnets, behind ALB
- **Launch Template** with the DuxxDB AMI/container, instance profile, user-data
- **Application Load Balancer** (+ target groups per service) · **ACM** cert · **WAF**
- (optional) **Network Load Balancer** for RESP (`6379`) if agents connect directly

### Data & storage
- **RDS for PostgreSQL** (Multi-AZ, gp3, KMS-encrypted) — control metadata
- **EBS gp3 volumes** (KMS-encrypted) — one per data-plane node = `--tenants-dir`
- **S3 buckets**: `duxx-objects` (document bytes), `duxx-snapshots` (replica bootstrap), `duxx-frontend` (SPA), `duxx-alb-logs`
- **CloudFront distribution** → `duxx-frontend` (OAC, ACM cert)

### Security & secrets
- **KMS** Customer-Managed Key(s) — EBS, RDS, S3, Secrets at-rest encryption (BYOK-ready)
- **Secrets Manager**: Ed25519 **JWT signing key** (private), RDS credentials, replication token, S3/object-store creds
- **IAM**: EC2 instance profile (least-privilege → S3, Secrets, KMS, CloudWatch), deploy role

### Observability & DNS
- **Route 53** hosted zone + records (`app.`, `api.`, `studio.`)
- **ACM** certificates (CloudFront in us-east-1; ALB in-region)
- **CloudWatch** Logs + Alarms; scrape `:9100` Prometheus metrics (self-managed Prometheus/Grafana or **Amazon Managed Prometheus**)

### Images / CI
- **ECR** repositories for `duxxdb` (data+control image) and `duxx-frontend`

---

## 4. EC2 sizing

DuxxDB's hot tier (HNSW vector index + tantivy) is **memory-resident**, so
data-plane nodes are **memory-optimized**:

| Tier | Instance | Use |
|---|---|---|
| Dev / single-box | `t3.large` (2 vCPU, 8 GB) | everything on one node |
| Data plane (prod) | `r6i.large`–`r6i.2xlarge` (16–64 GB) | per-tenant workspaces; size by total memories × dim |
| Control plane | `t3.medium` (stateless; scales horizontally) | API + JWT signing |
| RDS | `db.t3.medium` → `db.r6g.large` | control metadata (small, relational) |

**Rule of thumb:** ~`vectors × dim × 4 bytes` for HNSW + overhead. 1M chunks ×
1536-dim ≈ 6 GB just for vectors → an `r6i.xlarge` (32 GB) comfortably hosts
many such workspaces. Scale **out by tenant placement** (the control plane's
`place_project`) before scaling a single node up.

---

## 5. Networking & security posture

- **Private by default** — EC2 + RDS in private subnets, no public IPs. Only the
  ALB (and CloudFront) are public.
- **TLS everywhere** — ACM on ALB/CloudFront; DuxxDB also supports native
  TLS/mTLS on RESP (`--tls-cert/--tls-key`).
- **Auth** — clients present a control-plane **JWT**; the data plane verifies
  with the **public key only** (`--jwt-public-key`), so a node compromise can't
  mint tokens. Self-hosted/admin can use the static `--token`/`--auth-key`.
- **Encryption at rest** — KMS CMK on EBS, RDS, S3, Secrets (per-tenant keys are
  feasible thanks to physical workspace isolation).
- **Data residency** — pin a tenant's project to a node in the required region
  via `place_project`; one region's nodes + RDS never leave that region.
- **Audit** — every mutation is in the per-tenant audit trail (`/studio/audit`)
  + the JSON-lines `--audit-log` → ship to CloudWatch / S3.

---

## 6. Data, durability & backup

| Data | Lives in | Durability | Backup |
|---|---|---|---|
| Workspaces (memory/traces/evals/docs index) | EBS gp3 per node | redb ACID + HNSW/tantivy on disk | EBS snapshots + DuxxDB `--make-snapshot` → S3 |
| Control metadata | RDS Postgres | Multi-AZ + automated backups | RDS PITR |
| Document bytes | S3 | 11 nines | versioning + lifecycle/Glacier |
| Replication WAL | EBS (`--replication-wal`) | redb durable, compactable | covered by node snapshot |

**Replication / HA:** run a leader + follower(s) (`--replication-addr` /
`--replication-leader` / `--replication-token`); a fresh replica bootstraps from
a snapshot (`--bootstrap-snapshot`) then resumes the live feed. Add etcd +
`EtcdCoordinator` for automatic leader election/failover.

---

## 7. Single-EC2 quickstart (fastest path)

For a demo / small prod, run the whole stack on one box with Docker Compose
(`deploy/docker-compose.yml`):

```bash
# 1. Launch an EC2 instance (Amazon Linux 2023, r6i.large, 50 GB gp3),
#    SG allowing 443 (or 8080/7070/7072 for testing) from your IP.
# 2. Install Docker + compose plugin.
sudo dnf install -y docker && sudo systemctl enable --now docker
# 3. Clone + run:
git clone <repo> && cd duxxdb
export JWT_SECRET=$(openssl rand -hex 32)
docker compose -f deploy/docker-compose.yml up --build -d
```

Brings up: Postgres, MinIO, control (`:7070`), data plane (`:6379/:7072/:4318`),
and the React frontend (`:8080`). Put it behind an ALB + ACM for TLS, point
Route 53 at it, and you have a working single-node cloud.

---

## 8. Production deployment steps

1. **Build & push images** to ECR (`docker build` the root `Dockerfile` →
   `duxxdb:latest`; `deploy/Dockerfile.frontend` → `duxx-frontend`).
2. **Network**: VPC, subnets, IGW/NAT, SGs (Terraform/CloudFormation).
3. **Secrets**: generate the Ed25519 keypair (`duxx-control keygen`), store the
   **private** key in Secrets Manager (control reads it), distribute the
   **public** key to data-plane nodes (`--jwt-public-key`). Store RDS creds +
   replication token.
4. **Data stores**: RDS Postgres (Multi-AZ, KMS); S3 buckets (objects, snapshots,
   frontend) with KMS + versioning.
5. **Control plane**: ASG behind ALB (`/v1/*` target group), env
   `DUXX_CONTROL_ED25519_KEY` from Secrets, Postgres connection.
6. **Data plane**: ASG/fixed nodes with KMS-encrypted EBS at `--tenants-dir`,
   `--jwt-public-key`, `--studio-addr`/`--otlp-addr`, replication flags. Leader +
   followers across AZs.
7. **Frontend**: build with `VITE_CONTROL_URL`/`VITE_STUDIO_URL` (your API
   domain), sync `dist/` to S3, front with CloudFront + ACM.
8. **DNS/TLS**: Route 53 records → CloudFront/ALB; ACM certs.
9. **Observability**: CloudWatch agent for logs; scrape `:9100`; alarms on node
   health, RDS, replication lag.

---

## 9. Scaling & HA summary

- **Scale out by tenant** — the control plane's `place_project` spreads projects
  across data-plane nodes (shared nodes for small tenants, dedicated for large).
- **HA** — leader/follower replication + (with etcd) automatic failover; RDS
  Multi-AZ; ALB across AZs; EBS snapshots + DuxxDB snapshots to S3.
- **Reads** — lag-aware routing serves reads from fresh followers.
- **Frontend/control** — stateless, scale horizontally behind the ALB/CDN.

---

## 10. Rough monthly cost (small production, us-east-1)

| Resource | Spec | ~$/mo |
|---|---|---|
| Data-plane EC2 | 1× `r6i.xlarge` | ~$200 |
| Control EC2 | 1× `t3.medium` | ~$30 |
| RDS Postgres | `db.t3.medium` Multi-AZ | ~$120 |
| EBS gp3 | 100 GB + snapshots | ~$15 |
| S3 + CloudFront | light traffic | ~$10 |
| ALB + NAT + KMS + Secrets | baseline | ~$70 |
| **Total** | starter prod | **~$450/mo** |

Scale data-plane EC2 + EBS with tenant count; everything else grows slowly.
A single-EC2 quickstart (Section 7) runs for **~$120/mo**.
