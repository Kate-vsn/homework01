from datetime import UTC, datetime

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import CronTriggerTimetable, Param, dag, task
from kubernetes.client import models as k8s

DATA = "src/churn/data/housing.csv"
PULL = f"uv run --no-sync dvc pull -r cluster {DATA} && "


@dag(
    schedule=CronTriggerTimetable("0 3 * * *", timezone="Europe/Moscow"),
    catchup=False,
    params={
        "N_ESTIMATORS": Param(
            100, type="integer", minimum=10, description="деревьев в RandomForest"
        )
    },
    tags=["housing"],
)
def ml_retrain():
    def pod(task_id: str, name: str, cmd: str, **extra) -> KubernetesPodOperator:
        return KubernetesPodOperator(
            task_id=task_id,
            name=name,
            namespace="mlops",
            in_cluster=True,
            image="{{ var.value.ml_image }}",
            image_pull_policy="IfNotPresent",
            cmds=["sh", "-c", cmd],
            env_vars={
                "MLFLOW_TRACKING_URI": "http://mlflow.mlops:5000",
                **extra.pop("env_vars", {}),
            },
            env_from=[
                k8s.V1EnvFromSource(
                    secret_ref=k8s.V1SecretEnvSource(name="s3-credentials")
                )
            ],
            get_logs=True,
            on_finish_action="delete_pod",
            **extra,
        )

    validate = pod(
        "validate", "ml-validate", PULL + "uv run --no-sync python -m churn.data"
    )
    train = pod(
        "train",
        "ml-train",
        PULL + "uv run --no-sync python -m churn.train",
        env_vars={"N_ESTIMATORS": "{{ params.N_ESTIMATORS }}"},
        do_xcom_push=True,
    )

    @task.short_circuit
    def promoted(result: dict) -> bool:
        print(
            f"версия {result['version']}, r2 {result['r2']}, champion до этого {result['champion_before']}"
        )
        return result["promoted"]

    @task
    def rollout():
        from kubernetes import client, config

        config.load_incluster_config()
        now = datetime.now(UTC).isoformat()
        client.AppsV1Api().patch_namespaced_deployment(
            "ml-api-deployment",
            "default",
            {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {"kubectl.kubernetes.io/restartedAt": now}
                        }
                    }
                }
            },
        )
        print("ml-api-deployment перезапущен", now)

    validate >> train >> promoted(train.output) >> rollout()


ml_retrain()
