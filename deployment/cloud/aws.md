# AWS Deployment Guide

## Prerequisites

1. AWS CLI installed and configured
2. Docker installed
3. ECR repository created

## Deployment Steps

### 1. Create ECR Repository

```bash
aws ecr create-repository \
    --repository-name deepfake-detector \
    --image-scanning-configuration scanOnPush=true
```

### 2. Login to ECR

```bash
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

### 3. Build and Push Image

```bash
# Build
docker build -t deepfake-detector -f deployment/Dockerfile .

# Tag
docker tag deepfake-detector:latest \
    <account-id>.dkr.ecr.us-east-1.amazonaws.com/deepfake-detector:latest

# Push
docker push \
    <account-id>.dkr.ecr.us-east-1.amazonaws.com/deepfake-detector:latest
```

### 4. Deploy to ECS

#### Task Definition:
```json
{
  "family": "deepfake-detector",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/deepfake-detector:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/deepfake-detector",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Service Configuration:
- **Desired Count:** 2
- **Launch Type:** FARGATE
- **Cluster:** deepfake-cluster
- **Subnets:** Private subnets
- **Security Groups:** Allow inbound on port 8000

### 5. Set Up Load Balancer

```bash
# Create ALB
aws elbv2 create-load-balancer \
    --name deepfake-detector-alb \
    --subnets subnet-xxx subnet-yyy \
    --security-groups sg-xxx

# Create Target Group
aws elbv2 create-target-group \
    --name deepfake-detector-tg \
    --protocol HTTP \
    --port 8000 \
    --vpc-id vpc-xxx \
    --target-type ip

# Create Listener
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:... \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

## Cost Estimation

| Resource | Monthly Cost |
|----------|--------------|
| Fargate (2 tasks) | ~$30 |
| ALB | ~$18 |
| ECR Storage | ~$1 |
| CloudWatch Logs | ~$5 |
| **Total** | **~$54** |

## Monitoring

- CloudWatch Metrics for ECS
- ALB Access Logs
- Custom metrics via Prometheus (optional)

## Auto Scaling

```bash
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/deepfake-cluster/deepfake-detector \
    --min-capacity 1 \
    --max-capacity 10

aws application-autoscaling put-scaling-policy \
    --policy-name cpu-utilization \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/deepfake-cluster/deepfake-detector \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60
    }'
```
