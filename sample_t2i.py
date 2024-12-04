from pathlib import Path
from loguru import logger
from brdit.config import get_args
from brdit.inference import End2End

def inferencer():
    args = get_args()
    models_root_path = Path(args.model_root)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Load models
    gen = End2End(args, models_root_path)

    return args, gen

if __name__ == "__main__":
    args, gen = inferencer()

    # Ensure input image path is passed and valid
    if args.image_path is None:
        logger.error("No image path provided. Please specify an image for input.")
        raise ValueError("No image path provided.")

    # Run inference
    logger.info("Generating images...")
    height, width = args.image_size

    # Pass the image path as part of the prediction function
    results = gen.predict(args.prompt,
                          height=height,
                          width=width,
                          seed=args.seed,
                          enhanced_prompt=None,  # Removed T5 enhancement step
                          negative_prompt=args.negative,
                          infer_steps=args.infer_steps,
                          guidance_scale=args.cfg_scale,
                          batch_size=args.batch_size,
                          src_size_cond=args.size_cond,
                          use_style_cond=args.use_style_cond,
                          image_path=args.image_path  # Ensure the image path is passed
                          )

    images = results['images']

    # Save images
    save_dir = Path('results')
    save_dir.mkdir(exist_ok=True)

    cfg_scale = args.cfg_scale
    infer_steps = args.infer_steps
    for idx, pil_img in enumerate(images):
        # Ensure unique filename by incrementing idx if file exists
        save_path = save_dir / f"image_cfg{cfg_scale}_steps{infer_steps}_idx{idx}.png"
        while save_path.exists():
            idx += 1
            save_path = save_dir / f"image_cfg{cfg_scale}_steps{infer_steps}_idx{idx}.png"
        
        pil_img.save(save_path)
        logger.info(f"Saved image to {save_path}")
