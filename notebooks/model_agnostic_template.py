from pathlib import Path

from tracers.project import setup_project
from tracers.datasets.factory import build_datasets, build_loader_for_model
from tracers.pipelines.trainer_factory import build_trainer
from tracers.pipelines.evaluate import run_evaluation
from tracers.pipelines.generate import run_custom_generation, run_validation_generation
from tracers.export.chart import export_generated_chart

PROJECT_ROOT = Path.cwd().parent
setup = setup_project(PROJECT_ROOT)

MODEL_KEY = "tracer_np"  # or "tracer_global_np"
TASK = "evaluate"  # "evaluate", "generate_val", "generate_custom"

experiment_config = setup["experiment_config"]
hyperparams = setup["hyperparams"]
segment_frames = setup["segment_frames"]
batch_size = setup["batch_size"]
device = setup["device"]

if TASK in {"evaluate", "generate_val"}:
    _, _, train_dataset, val_dataset, test_dataset, *_ = build_datasets(
        PROJECT_ROOT,
        segment_frames=segment_frames,
        seed=42,
    )

    train_loader = build_loader_for_model(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        model_key=MODEL_KEY,
    )
    val_loader = build_loader_for_model(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        model_key=MODEL_KEY,
    )
else:
    train_loader = None
    val_loader = None

checkpoint_path = str(PROJECT_ROOT / "notebooks" / "checkpoints" / "best.pt")
trainer, model = build_trainer(
    model_key=MODEL_KEY,
    hyperparams=hyperparams,
    experiment_config=experiment_config,
    device=device,
    checkpoint_path=checkpoint_path if Path(checkpoint_path).exists() else None,
    train_loader=train_loader,
    valid_loader=val_loader,
)

if TASK == "evaluate":
    run_evaluation(trainer, val_loader, checkpoint=checkpoint_path, name="VAL")
elif TASK == "generate_val":
    CHART_ID = None
    context_frames = 500
    generated, actual, debug_segments = run_validation_generation(
        trainer,
        val_loader,
        chart_id=CHART_ID,
        context_frames=context_frames,
        seed_action=None,
        temperature=None,
        top_k=None,
        top_p=None,
        seed=0,
        collect_debug=True,
    )
    out_dir = Path("charts") / str(CHART_ID)
    export_generated_chart(generated, out_dir / "generated_chart.json", bpm=120, start_frame=0)
    export_generated_chart(actual, out_dir / "actual_chart.json", bpm=120, start_frame=0)
elif TASK == "generate_custom":
    AUDIO_PT = PROJECT_ROOT / "renai.pt"
    generated, debug_segments = run_custom_generation(
        trainer,
        AUDIO_PT,
        difficulty=5,
        segment_frames=segment_frames,
        context_frames=500,
        seed_action=0,
        seed=0,
        temperature=0.4,
        top_k=None,
        top_p=None,
        collect_debug=False,
    )
    out_dir = Path("charts") / "renai"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_generated_chart(generated, out_dir / "generated_chart.json", bpm=120, start_frame=0)
else:
    raise ValueError(f"Unknown TASK: {TASK}")
