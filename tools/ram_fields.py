def ram_fields():
    cell_fields = [
        "Density",
        "x-velocity",
        "y-velocity",
        "z-velocity",
        "Pressure",
        "Metallicity",
        # "dark_matter_density",
        "xHI",
        "xHII",
        "xHeII",
        "xHeIII",
    ]
    epf = [
        ("particle_family", "b"),
        ("particle_tag", "b"),
        ("particle_birth_epoch", "d"),
        ("particle_metallicity", "d"),
    ]
    return cell_fields, epf
