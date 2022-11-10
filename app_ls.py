from lai_components.label_studio.component import LitLabelStudio

import lightning.app as la


class LitApp(la.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.lit_label_studio = LitLabelStudio()

    def run(self):
        self.lit_label_studio.run()

    def configure_layout(self):
        return {"name": "Annotate", "content": self.lit_label_studio.label_studio}


app = la.LightningApp(LitApp())
