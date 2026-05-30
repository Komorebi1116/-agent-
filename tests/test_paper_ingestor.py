from vs_agent.ingestion.paper_ingestor import filename_to_title


def test_filename_to_title_preserves_symbols_and_keywords():
    filename = "Dense_pix2pix-CycleGAN_CSI-GAN_ViT-Stain_H&E.pdf"

    title = filename_to_title(filename)

    assert title == "Dense_pix2pix-CycleGAN_CSI-GAN_ViT-Stain_H&E"
