import shutil, tempfile, os

class BestKeeper:
    def __init__(self, save_model_path, minimise):
        self.best_score = None
        self.temp_dir = tempfile.TemporaryDirectory()
        self.best_temp = os.path.join(self.temp_dir.name, "bestmodel.safetensors")
        self.save_model_path = save_model_path
        self.minimise = minimise

    def keep_if_best(self, score):
        if self.best_score is None or (score<self.best_score and self.minimise) or (score>self.best_score and not self.minimise):
            shutil.copyfile(self.save_model_path, self.best_temp)
            self.best_score = score

    def restore_best(self):
        assert self.best_score is not None
        shutil.copyfile(self.best_temp, self.save_model_path)
        return self.save_model_path

    def bad_by(self, score, margin, absolute):
        if score is None: return True
        if absolute is not None:
            if (score > absolute and self.minimise) or (score < absolute and not self.minimise): return True

        if self.best_score is not None and margin is not None:
            if (score > self.best_score + margin and self.minimise) or (score < self.best_score - margin and not self.minimise): return True

        return False