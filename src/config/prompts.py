# src/config/prompts.py
"""
Centralización de prompts del sistema e instrucciones para los LLMs.
Facilita el mantenimiento, la localización y el escalado de prompts.
"""


class SystemPrompts:
    # --- CHAT ORCHESTRATOR ---
    DEFAULT_SYSTEM_PROMPT = "Eres un asistente de IA que responde siempre en español."

    HALLUCINATION_STRICT = (
        "\n\nIMPORTANTE: NUNCA debes inventar información. Responde únicamente con hechos "
        "verificables del contexto proporcionado. Cita siempre tus fuentes. Si no tienes la "
        "información, dilo claramente."
    )

    HALLUCINATION_CREATIVE = (
        "\n\nPuedes hacer inferencias y sugerencias razonables. Si estás especulando, indícalo claramente."
    )

    FILE_REFERENCE_INSTRUCTIONS = (
        "\n\nCuando la información provenga de documentos proporcionados por una herramienta, "
        "cita los archivos usando el formato [Archivo: nombre.ext].\n\n"
        "Cuando la información provenga de una API o herramienta externa que no contiene documentos, "
        "NO cites archivos."
    )

    SOURCE_CITATION_INSTRUCTIONS = (
        "\n\nNunca inventes documentos o fuentes que no existan.\n\n"
        "Si el usuario hace referencia a 'esto', 'ese archivo', 'lo que envié', etc., "
        "y hay archivos en el contexto de herramientas, asume que se refiere a ellos."
    )
    
    # Concise header for tool results - optimized for efficiency
    SOURCE_OF_TRUTH_HEADER = (
        "## CONTEXTO DE HERRAMIENTAS"
        "\nInformación verificada de herramientas. Responde según lo solicitado.\n\n"
    )

    # Header for RAG tool - includes collection references
    SOURCE_OF_TRUTH_RAG = (
        "## DOCUMENTACIÓN (BÚSQUEDA RAG)\n"
        "Información de archivos en colecciones. Usa esta información para responder. "
        "Cita las fuentes [Archivo: nombre] en tu respuesta.\n\n"
    )

    # Ultra-concise alternative (can be used for simpler responses)
    SOURCE_OF_TRUTH_CONCISE = (
        "## Resultado de herramienta\n"
        "Usa esta información para responder.\n\n"
    )

    # Concise header for codebase tool
    SOURCE_OF_TRUTH_CODEBASE = (
        "## ANÁLISIS DE CÓDIGO\n"
        "Información verificada del código. Responde exactamente lo que el usuario preguntó. "
        "Si pregunta 'cuántos métodos tiene', responde CON EL NÚMERO. "
        "Sé conciso.\n\n"
    )

    # --- CÓDIGO FUENTE EN CONTEXTO RAG ---
    # Patrón regex multi-lenguaje para detectar firmas de funciones/métodos/clases.
    # Añadir aquí nuevos lenguajes; context_builder.py NO necesita cambios.
    #
    # Lenguajes cubiertos:
    #   Python      → def foo(, async def foo(, class Foo
    #   JS/TS       → function foo(, const/let/var foo = async (, export [default] [async] function
    #   Java/Kotlin → public/private/protected/static ... foo(
    #   C#          → public/private/protected/internal/static/abstract/override/virtual/sealed ... foo(
    #   Go          → func foo(
    #   Rust        → fn foo(, pub fn foo(
    #   PHP         → function foo(, public|private|protected function foo(
    #   Ruby/Swift  → def foo
    CODE_SIGNATURE_PATTERN = (
        r"(?m)(?:"
        r"\basync\s+def\s+\w+\s*\(|\bdef\s+\w+\s*\(|\bclass\s+\w+"               # Python
        r"|\bfunction\s+\w+\s*\("                                                   # JS/TS named func
        r"|\b(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\("                       # JS/TS arrow
        r"|\b(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b"                 # JS/TS export
        r"|\b(?:public|private|protected|internal|static|abstract"
        r"|override|virtual|sealed)\s+(?:\w+\s+)*\w+\s*\("                        # Java/Kotlin/C#
        r"|\bfunc\s+\w+\s*\("                                                      # Go / Swift
        r"|\b(?:pub\s+)?fn\s+\w+\s*\("                                            # Rust
        r")"
    )

    CODE_ANTI_HALLUCINATION_INSTRUCTION = (
        "⚠️ **INSTRUCCIÓN CRÍTICA - MÁXIMA PRIORIDAD:** "
        "Los chunks siguientes contienen código REAL y EXACTO del archivo fuente. "
        "CUANDO EL USUARIO PIDE CÓDIGO DE UNA FUNCIÓN/MÉTODO/CLASE, "
        "**DEBES DEVOLVER EL CÓDIGO EXACTO DE LOS CHUNKS SIN MODIFICARLO**. "
        "NUNCA reescribas, regeneres o inventes código de tu memoria. "
        "Si los chunks no tienen el código completo, responde exactamente: 'Código no encontrado en los chunks disponibles'. "
        "Usa formato markdown con el bloque de código textualmente."
        ""
    )


    # Concise footer
    SOURCE_OF_TRUTH_FOOTER = ""

    # For basic_analyze_file (structure only)
    SOURCE_OF_TRUTH_BASIC_STRUCTURE = (
        "## ANÁLISIS ESTRUCTURAL\n"
        "Responde únicamente la información solicitada. Sé conciso.\n\n"
    )

    # For analyze_file (full analysis with quality metrics)
    SOURCE_OF_TRUTH_FULL_ANALYSIS = (
        "## ANÁLISIS COMPLETO\n"
        "Resume hallazgos principales. Sé conciso.\n\n"
    )

    # --- EXTRACTION SERVICE ---
    # Template genérico para extracción de parámetros de custom tools
    EXTRACTION_GENERIC_TEMPLATE = """Eres un bot de extracción de datos de alto rendimiento.
    Tu ÚNICO trabajo es extraer variables del MENSAJE DE USUARIO en formato JSON.

    ### VARIABLES A EXTRAER:
    {params_desc}

    ### REGLAS CRÍTICAS:
    1. Retorna ÚNICAMENTE un objeto JSON puro. Sin conversación, sin explicaciones, sin etiquetas de markdown adicionales.
    2. **NUNCA retornes valores null, None, ni strings vacíos "". Si no hay valor, simplemente NO incluyas la llave en el JSON.**
    3. Si una variable NO se encuentra en el mensaje de usuario, NO la incluyas en el JSON final.
    4. **LLAVES DUPLICADAS/SIMILARES**: Si hay múltiples llaves (ej. 'nombre_ciudad' y 'nom_ciudad') que parecen referirse al mismo concepto semántico, asigna el valor extraído a TODAS ellas.
    5. Usa los nombres exactos definidos arriba como LLAVES (KEYS).
    6. Si un parámetro tiene VALORES PERMITIDOS, usa ÚNICAMENTE esos valores. Si el valor en el mensaje no coincide exactamente pero es semánticamente igual, usa el valor permitido correspondiente.
    7. Si no puedes extraer NINGUNA de las variables solicitadas con seguridad, retorna un objeto vacío {{}}.

    ### EJEMPLOS DE SALIDA:
    {{"ciudad": "Lima"}}
    {{"nombre_ciudad": "Madrid", "pais": "España"}}
    {{"temperatura": "25"}}
    {{}}
    """

    EXTRACTION_USER_MESSAGE_TEMPLATE = "MENSAJE DE USUARIO: {user_message}\n\nExtrae únicamente: {expected_keys}. Retorna solo JSON."
    EXTRACTION_PREVIOUS_CONTEXT_LABEL = "CONTEXTO PREVIO:"
    EXTRACTION_NO_DESCRIPTION = "No hay descripción disponible"
    EXTRACTION_REQUIRED_LABEL = " (REQUERIDO)"
    EXTRACTION_ALLOWED_VALUES_LABEL = " (VALORES PERMITIDOS: {enum})"

    # --- MEMORY SERVICE ---
    MEMORY_CONTEXT_HEADER = "## Contexto relevante de conversaciones pasadas\n\n"
    MEMORY_MESSAGE_FORMAT = "[{i}] **{role}** (relevancia: {score:.2f}, {timestamp}):\n{content}\n\n"

    # --- INTENT ANALYZER ---
    INTENT_REASONING_UP = "La consulta busca información contextual/general"
    INTENT_REASONING_DOWN = "La consulta busca detalles/instancias específicas"
    INTENT_REASONING_BOTH = "La consulta busca información exhaustiva/completa"
    INTENT_REASONING_STARTING_NOTE = "Partiendo de una nota de tipo {note_type}"
    INTENT_REASONING_DEFAULT = "No se detectó un patrón claro - usando búsqueda bidireccional segura"
    INTENT_REASONING_HEURISTIC = "Heurística por defecto aplicada"
