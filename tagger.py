import gradio as gr
from models_list import models, SWINV2_MODEL_DSV3_REPO
from predictor import Predictor
from utility import modify_files_in_directory, search_and_replace, remove_duplicates, resize_images
import deepdanbooru as dd


def main():
    predictor = Predictor()
    with gr.Blocks(title='Image Tagging utility') as iface:
        with gr.Tab("Interrogator"):
            with gr.Column():
                gr.Markdown(value=f"<h2>Generate prompt for image</h2>")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        image = gr.Image(type="pil", image_mode="RGBA", label="Input")
                        model_repo = gr.Dropdown(
                            models,
                            value=SWINV2_MODEL_DSV3_REPO,
                            label="Model",
                        )
                        with gr.Row():
                            general_thresh = gr.Slider(
                                0,
                                1,
                                step=0.05,
                                value=0.35,
                                label="General Tags Threshold",
                                scale=3,
                            )
                            general_mcut_enabled = gr.Checkbox(
                                value=False,
                                label="Use MCut threshold",
                                scale=1,
                            )

                        with gr.Row():
                            character_thresh = gr.Slider(
                                0,
                                1,
                                step=0.05,
                                value=0.85,
                                label="Character Tags Threshold",
                                scale=3,
                            )
                            character_mcut_enabled = gr.Checkbox(
                                value=False,
                                label="Use MCut threshold",
                                scale=1,
                            )

                        with gr.Row():
                            clear = gr.ClearButton(
                                components=[
                                    image,
                                    model_repo,
                                    general_thresh,
                                    general_mcut_enabled,
                                    character_thresh,
                                    character_mcut_enabled,
                                ],
                                variant='secondary',
                                size='lg',
                            )
                            submit = gr.Button(value='Submit', variant='primary', size='lg')

                    with gr.Column(variant='panel'):
                        sorted_general_strings = gr.Textbox(label='Output string')
                        rating = gr.Label(label='Rating')
                        character_res = gr.Label(label='Output (characters)')
                        general_res = gr.Label(label='Output (tags)')
                        clear.add(
                            [
                                sorted_general_strings,
                                rating,
                                character_res,
                                general_res,
                            ]
                        )

            submit.click(
                fn=predictor.predict,
                inputs=[
                    image,
                    model_repo,
                    general_thresh,
                    general_mcut_enabled,
                    character_thresh,
                    character_mcut_enabled,
                ],
                outputs=[sorted_general_strings, rating, character_res, general_res]
            )

        with gr.Tab("Tagging"):
            with gr.Column():
                with gr.Tab("WD Tagger"):
                    gr.Markdown(value=f"<h2>Tag images</h2>")
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            dir = gr.Textbox(label='Input Directory')
                            model_repo = gr.Dropdown(
                                models,
                                value=SWINV2_MODEL_DSV3_REPO,
                                label="Model",
                            )
                            with gr.Row():
                                general_thresh = gr.Slider(
                                    0,
                                    1,
                                    step=0.05,
                                    value=0.35,
                                    label="General Tags Threshold",
                                    scale=3,
                                )
                                general_mcut_enabled = gr.Checkbox(
                                    value=False,
                                    label="Use MCut threshold",
                                    scale=1,
                                )

                            with gr.Row():
                                character_thresh = gr.Slider(
                                    0,
                                    1,
                                    step=0.05,
                                    value=0.85,
                                    label="Character Tags Threshold",
                                    scale=3,
                                )
                                character_mcut_enabled = gr.Checkbox(
                                    value=False,
                                    label="Use MCut threshold",
                                    scale=1,
                                )
                            with gr.Row():
                                append_tags = gr.Checkbox(value=False, label="Append Tags",
                                                          info="Do you want to append tags to existing files?")

                            with gr.Row():
                                clear = gr.ClearButton(
                                    components=[
                                        dir,
                                        model_repo,
                                        general_thresh,
                                        general_mcut_enabled,
                                        character_thresh,
                                        character_mcut_enabled,
                                        append_tags
                                    ],
                                    variant='secondary',
                                    size='lg',
                                )
                                submit = gr.Button(value='Submit', variant='primary', size='lg')

                        with gr.Column(variant='panel'):
                            sorted_general_strings = gr.Textbox(label='Output string')
                            general_res = gr.HTML(label='Output (tags)')
                            clear.add(
                                [
                                    sorted_general_strings,
                                    general_res,
                                ]
                            )
                submit.click(
                    fn=predictor.label_images,
                    inputs=[
                        dir,
                        model_repo,
                        general_thresh,
                        general_mcut_enabled,
                        character_thresh,
                        character_mcut_enabled,
                        append_tags
                    ],
                    outputs=[sorted_general_strings, general_res]
                )
                with gr.Tab("DeepDanbooru Tagger"):
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            dir_name = gr.Textbox(label='Input Directory')
                            with gr.Row():
                                score_thresh = gr.Slider(
                                    0,
                                    1,
                                    step=0.05,
                                    value=0.5,
                                    label="Score Threshold",
                                )

                            with gr.Row():
                                clear = gr.ClearButton(
                                    components=[
                                        dir_name,
                                        score_thresh,
                                    ],
                                    variant='secondary',
                                    size='lg',
                                )
                                submit = gr.Button(value='Submit', variant='primary', size='lg')

                        with gr.Column(variant='panel'):
                            sorted_general_strings = gr.Textbox(label='Output string')
                            general_res = gr.HTML(label='Output (tags)')
                            clear.add(
                                [
                                    sorted_general_strings,
                                    general_res,
                                ]
                            )

                submit.click(
                    fn=predictor.predict_deepbooru,
                    inputs=[
                        dir_name,
                        score_thresh,
                    ],
                    outputs=[sorted_general_strings, general_res]
                )

        with gr.Tab('Append/Prepend Tags'):
            with gr.Column():
                gr.Markdown(value=f"<h2>Generate prompt for image</h2>")
                with gr.Row():
                    with gr.Column(variant="panel"):
                        dpath = gr.Textbox(label="Enter the path to the caption files directory")
                        ptext = gr.Textbox(label="Enter the text to append/prepend")
                        append = gr.Checkbox(label="Prepend tags?", value=True)
                    with gr.Column(variant='panel'):
                        output = gr.Textbox(label='Status')

            btn = gr.Button(value='Submit', variant='primary', size='lg')
            btn.click(fn=modify_files_in_directory, inputs=[dpath, ptext, append], outputs=output)

        with gr.Tab('Search/Replace Tag'):
            with gr.Row():
                dpath = gr.Textbox(label="Enter the path to the caption files directory")
                search = gr.Textbox(label="Text to search")
                replace = gr.Textbox(label="Text to replace with")
                output = gr.Textbox(label='Status')
            btn = gr.Button(value='Submit', variant='primary', size='lg')
            btn.click(fn=search_and_replace, inputs=[dpath, search, replace], outputs=output)

        with gr.Tab('Remove Duplicate Tags'):
            with gr.Row():
                dir_name = gr.Textbox(label='Input Directory')
                output = gr.Textbox(label='Status')
            btn = gr.Button(value='Submit', variant='primary', size='lg')
            btn.click(fn=remove_duplicates, inputs=[dir_name], outputs=output)

        with gr.Tab('Image Resizer'):
            with gr.Row():
                input_dir = gr.Textbox(label="Enter the path to the files directory")
            with gr.Row():
                output_dir = gr.Textbox(label="Enter the path to the files directory")
            with gr.Row():
                width = gr.Slider(100, 2000, step=50, label="Width", value=512)
                height = gr.Slider(100, 2000, step=50, label="Height", value=768)
                output = gr.Textbox(label='Status')
            btn = gr.Button(value='Submit', variant='primary', size='lg')
            btn.click(fn=resize_images, inputs=[input_dir, output_dir, width, height], outputs=output)

    iface.queue(max_size=1)
    iface.launch(debug=True)


if __name__ == '__main__':
    main()
