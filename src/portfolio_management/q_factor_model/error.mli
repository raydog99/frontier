exception DataError of string
exception ModelError of string
exception StatisticalError of string
exception VisualizationError of string

val handle_error : (unit -> unit) -> unit