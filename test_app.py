from lightning.app import LightningApp, LightningFlow, LightningWork
import os
import yaml


class LitPoseApp(LightningFlow):

    def __init__(self):
        super().__init__()
        self.proj_dir = os.path.join('data/test0')
        self.label_studio = LitLabelStudio()

    def run(self):
        self.label_studio.run(action="update_paths", proj_dir=self.proj_dir)
        self.label_studio.run(action="create_labeling_config_xml")

    def configure_layout(self):
        annotate_tab = {"name": "Label Frames", "content": self.label_studio}
        return [annotate_tab]


class LitLabelStudio(LightningWork):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.proj_dir = None
        self.test_str = 'this is a test string'
        self.filenames = {"label_studio_config": ""}

    def _update_paths(self, proj_dir):
        self.proj_dir = proj_dir
        self.filenames["label_studio_config"] = os.path.join(self.proj_dir, 'myfile.txt')
        print(self.filenames)

    def _create_labeling_config_xml(self):
        """Create a label studio configuration xml file."""
        proj_dir = os.path.abspath(self.proj_dir)
        print(self.filenames)
        print(self.test_str)
        config_file = os.path.join(proj_dir, os.path.basename(self.filenames['label_studio_config']))
        os.makedirs(proj_dir, exist_ok=True)
        with open(config_file, 'wt') as outfile:
            outfile.write('TESTING')

    def run(self, action=None, **kwargs):
        if action == "create_labeling_config_xml":
            self._create_labeling_config_xml(**kwargs)
        elif action == "update_paths":
            self._update_paths(**kwargs)

app = LightningApp(LitPoseApp())
