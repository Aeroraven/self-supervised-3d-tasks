from self_supervised_3d_tasks.train import main
from self_supervised_3d_tasks.adaptive.gpu_adapt import  gpu_autogrow

if __name__ == "__main__":
    gpu_autogrow()
    main()
