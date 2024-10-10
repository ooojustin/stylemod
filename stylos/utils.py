def get_full_module_path(variable):
    variable_type = type(variable)
    module = variable_type.__module__
    class_name = variable_type.__name__
    return f"{module}.{class_name}"
