using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.ML.OnnxRuntime.Tensors;

class Program
{
    private static InferenceSession session;
    private static Tokenizer tokenizer;
    private static Model model;
    
    // Token IDs for "yes" and "no"
    private static int yesTokenId;
    private static int noTokenId;
    private static int yesTokenId2; // with space
    private static int noTokenId2; // with space
    
    private static readonly string systemPrompt =
        "Judge whether the JobDescription meets the requirements based on the Vacancy. Note that the answer can only be \"yes\" or \"no\".";

    static void Main(string[] args)
    {
        var modelPath = @"C:\repos\Phi-4-mini-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4\";
        
        InitializeModel(modelPath);
        InitializeTokenIds();
        
        RunTests();
        
        // Cleanup
        session?.Dispose();
        tokenizer?.Dispose();
        model?.Dispose();
    }
    
    static void InitializeModel(string modelPath)
    {
        // Initialize model and tokenizer for token encoding
        model = new Model(modelPath);
        tokenizer = new Tokenizer(model);
        
        // Create ONNX Runtime session
        var sessionOptions = new SessionOptions();
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        
        // Look for the ONNX model file in the directory
        var onnxModelPath = System.IO.Path.Combine(modelPath, "model.onnx");
        if (!System.IO.File.Exists(onnxModelPath))
        {
            // Try alternative names
            var possibleNames = new[] { "phi-4.onnx", "model_quantized.onnx", "model.onnx", "decoder_model.onnx" };
            foreach (var name in possibleNames)
            {
                var path = System.IO.Path.Combine(modelPath, name);
                if (System.IO.File.Exists(path))
                {
                    onnxModelPath = path;
                    break;
                }
            }
        }
        
        if (!System.IO.File.Exists(onnxModelPath))
        {
            throw new FileNotFoundException($"Could not find ONNX model file in {modelPath}. Tried: model.onnx, phi-4.onnx, model_quantized.onnx, decoder_model.onnx");
        }
        
        session = new InferenceSession(onnxModelPath, sessionOptions);
        
        Console.WriteLine("Model initialized successfully");
        Console.WriteLine($"ONNX model path: {onnxModelPath}");
        Console.WriteLine($"Input names: {string.Join(", ", session.InputMetadata.Keys)}");
        Console.WriteLine($"Output names: {string.Join(", ", session.OutputMetadata.Keys)}");
        
        // Print detailed input metadata
        Console.WriteLine("\nDetailed input metadata:");
        foreach (var input in session.InputMetadata)
        {
            Console.WriteLine($"  {input.Key}: shape=[{string.Join(", ", input.Value.Dimensions)}], type={input.Value.ElementType}");
        }
    }
    
    static void InitializeTokenIds()
    {
        // Get token IDs for "yes" and "no"
        var yesTokens = tokenizer.Encode("Yes");
        var noTokens = tokenizer.Encode("No");
        var yesTokens2 = tokenizer.Encode(" yes");
        var noTokens2 = tokenizer.Encode(" no");
        var yesTokens3 = tokenizer.Encode("Yes");
        var noTokens3 = tokenizer.Encode("No");
        
        // Extract single token IDs
        yesTokenId = yesTokens[0][0];
        noTokenId = noTokens[0][0];
        yesTokenId2 = yesTokens2[0][0];
        noTokenId2 = noTokens2[0][0];
        
        yesTokenId2 = yesTokens3[0][0];
        noTokenId2 = noTokens3[0][0];

        Console.WriteLine($"Yes token ID: {yesTokenId}");
        Console.WriteLine($"No token ID: {noTokenId}");
        Console.WriteLine($"Yes token ID (with space): {yesTokenId2}");
        Console.WriteLine($"No token ID (with space): {noTokenId2}");
        Console.WriteLine();
    }
    
    static float CalculateYesProbability(float[] logits, int vocabSize)
    {
        // For single inference, logits should be [vocab_size]
        // Get logits for "yes" and "no" tokens
        if (logits.Length < Math.Max(Math.Max(yesTokenId, noTokenId), Math.Max(yesTokenId2, noTokenId2)))
        {
            throw new ArgumentException($"Logits array too small. Expected at least {Math.Max(Math.Max(yesTokenId, noTokenId), Math.Max(yesTokenId2, noTokenId2))} elements, got {logits.Length}");
        }
        
        float yesLogit = logits[yesTokenId];
        float noLogit = logits[noTokenId];
        float yesLogit2 = logits[yesTokenId2];
        float noLogit2 = logits[noTokenId2];

        Console.WriteLine($"Yes logit: {yesLogit:F4}, No logit: {noLogit:F4}");
        Console.WriteLine($"Yes logit2 (with space): {yesLogit2:F4}, No logit2 (with space): {noLogit2:F4}");
        
        // Use the version without space prefix (more direct)
        // Apply softmax normalization for numeric stability
        float maxLogit = Math.Max(yesLogit, noLogit);
        float expYes = (float)Math.Exp(yesLogit - maxLogit);
        float expNo = (float)Math.Exp(noLogit - maxLogit);
        float sumExp = expYes + expNo;
        
        float yesProbability = expYes / sumExp;
        
        // Also calculate probabilities for space-prefixed versions for comparison
        float maxLogit2 = Math.Max(yesLogit2, noLogit2);
        float expYes2 = (float)Math.Exp(yesLogit2 - maxLogit2);
        float expNo2 = (float)Math.Exp(noLogit2 - maxLogit2);
        float sumExp2 = expYes2 + expNo2;
        float yesProbability2 = expYes2 / sumExp2;
        
        Console.WriteLine($"Yes probability (no space): {yesProbability:F4}");
        Console.WriteLine($"Yes probability (with space): {yesProbability2:F4}");
        
        return yesProbability;
    }
    
    static string FormatInstruction(string instruction, string query, string document)
    {
        if (string.IsNullOrEmpty(instruction))
            instruction = "Given a vacancy title, retrieve relevant job description of the candidate that is suitable for the vacancy";
        
        return $"Vacancy:\"\n {query} \"\n JobDescription:\"\n {document} \"\n";
    }
    
    static NamedOnnxValue CreateEmptyKVCache(string inputName, NodeMetadata metadata)
    {
        var originalDims = metadata.Dimensions.ToArray();
        Console.WriteLine($"Creating KV cache for {inputName} with original dims: [{string.Join(", ", originalDims)}]");
        
        var dims = new int[originalDims.Length];
        
        for (int i = 0; i < originalDims.Length; i++)
        {
            if (originalDims[i] == -1)
            {
                // Common patterns for KV cache dimensions:
                // [batch_size, num_heads, seq_len, head_dim]
                if (i == 0) dims[i] = 1;      // batch_size
                else if (i == 1) dims[i] = 8;  // num_heads (from error log)
                else if (i == 2) dims[i] = 0;  // seq_len (empty for initial cache)
                else if (i == 3) dims[i] = 128; // head_dim
                else dims[i] = 1;
            }
            else
            {
                dims[i] = (int)originalDims[i];
            }
        }
        
        // Ensure sequence dimension (usually index 2) is 0 for empty cache
        if (dims.Length >= 3)
        {
            dims[2] = 0;
        }
        
        Console.WriteLine($"Adjusted dims for {inputName}: [{string.Join(", ", dims)}]");
        
        // Calculate total elements - handle the case where dimensions multiply to 0
        long totalElements = dims.Aggregate(1L, (a, b) => a * b);
        Console.WriteLine($"Total elements needed: {totalElements}");
        
        // If total elements is 0, create an empty tensor properly
        if (totalElements == 0)
        {
            // Create empty tensor with proper dimensions
            var emptyArray = new float[0];
            var emptyCache = new DenseTensor<float>(emptyArray, dims);
            return NamedOnnxValue.CreateFromTensor(inputName, emptyCache);
        }
        else
        {
            // Create tensor with calculated number of elements
            var cacheArray = new float[totalElements];
            var cache = new DenseTensor<float>(cacheArray, dims);
            return NamedOnnxValue.CreateFromTensor(inputName, cache);
        }
    }
    
    static (float probability, string reasoning) ProcessPair(string query, string document)
    {
        try
        {
            // Format the prompt
            string formattedInput = FormatInstruction("", query, document);
            var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{formattedInput}<|end|><|assistant|>";
            
            Console.WriteLine($"Full prompt: {fullPrompt}");
            
            // Tokenize the input
            var tokens = tokenizer.Encode(fullPrompt);
            var inputIds = tokens[0].ToArray();
            
            Console.WriteLine($"Input token count: {inputIds.Length}");
            
            // Prepare input tensors for ONNX session
            var inputTensors = new List<NamedOnnxValue>();
            
            // Add required inputs based on model metadata
            foreach (var inputMeta in session.InputMetadata)
            {
                switch (inputMeta.Key.ToLower())
                {
                    case "input_ids":
                        var inputIdsTensor = new DenseTensor<long>(
                            inputIds.Select(x => (long)x).ToArray(), 
                            new int[] { 1, inputIds.Length });
                        inputTensors.Add(NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor));
                        Console.WriteLine($"Added input_ids: [1, {inputIds.Length}]");
                        break;
                        
                    case "attention_mask":
                        var attentionMask = new DenseTensor<long>(
                            Enumerable.Repeat(1L, inputIds.Length).ToArray(), 
                            new int[] { 1, inputIds.Length });
                        inputTensors.Add(NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask));
                        Console.WriteLine($"Added attention_mask: [1, {inputIds.Length}]");
                        break;
                        
                    case "position_ids":
                        var positionIds = new DenseTensor<long>(
                            Enumerable.Range(0, inputIds.Length).Select(x => (long)x).ToArray(), 
                            new int[] { 1, inputIds.Length });
                        inputTensors.Add(NamedOnnxValue.CreateFromTensor("position_ids", positionIds));
                        Console.WriteLine($"Added position_ids: [1, {inputIds.Length}]");
                        break;
                        
                    default:
                        // Handle KV cache and other inputs
                        if (inputMeta.Key.Contains("past_key_values") || inputMeta.Key.Contains("cache"))
                        {
                            var kvCache = CreateEmptyKVCache(inputMeta.Key, inputMeta.Value);
                            inputTensors.Add(kvCache);
                        }
                        else
                        {
                            Console.WriteLine($"Warning: Unknown input '{inputMeta.Key}' - skipping");
                        }
                        break;
                }
            }
            
            Console.WriteLine($"Total input tensors prepared: {inputTensors.Count}");
            
            // Run inference
            using var results = session.Run(inputTensors);
            
            // Get logits from output
            var logitsOutput = results.FirstOrDefault(r => r.Name.Contains("logits") || r.Name == "output" || r.Name.Contains("prediction"));
            if (logitsOutput == null)
            {
                // Try the first output if no obvious logits output
                logitsOutput = results.First();
                Console.WriteLine($"Using output: {logitsOutput.Name}");
            }
            
            var logitsTensor = logitsOutput.AsTensor<float>();
            var logitsShape = logitsTensor.Dimensions.ToArray();
            
            Console.WriteLine($"Logits tensor shape: [{string.Join(", ", logitsShape)}]");
            
            // Extract logits for the last token position
            float[] lastTokenLogits;
            
            if (logitsShape.Length == 3) // [batch_size, sequence_length, vocab_size]
            {
                int batchSize = logitsShape[0];
                int seqLength = logitsShape[1];
                int vocabSizeInIf = logitsShape[2];
                
                // Get logits for the last position
                lastTokenLogits = new float[vocabSizeInIf];
                for (int i = 0; i < vocabSizeInIf; i++)
                {
                    lastTokenLogits[i] = logitsTensor[0, seqLength - 1, i];
                }
            }
            else if (logitsShape.Length == 2) // [batch_size, vocab_size]
            {
                int vocabSizeInIf = logitsShape[1];
                lastTokenLogits = new float[vocabSizeInIf];
                for (int i = 0; i < vocabSizeInIf; i++)
                {
                    lastTokenLogits[i] = logitsTensor[0, i];
                }
            }
            else
            {
                throw new InvalidOperationException($"Unexpected logits tensor shape: [{string.Join(", ", logitsShape)}]");
            }
            
            int vocabSize = lastTokenLogits.Length;
            Console.WriteLine($"Vocab size: {vocabSize}");
            
            // Calculate yes probability
            float yesProbability = CalculateYesProbability(lastTokenLogits, vocabSize);
            
            // Get the top predicted token for reasoning
            int maxTokenId = 0;
            float maxLogit = lastTokenLogits[0];
            for (int i = 1; i < lastTokenLogits.Length; i++)
            {
                if (lastTokenLogits[i] > maxLogit)
                {
                    maxLogit = lastTokenLogits[i];
                    maxTokenId = i;
                }
            }
            
            // Decode the predicted token
            var predictedTokens = new int[] { maxTokenId };
            var decodedText = tokenizer.Decode(predictedTokens);
            
            return (yesProbability, $"Most likely next token: '{decodedText}' (ID: {maxTokenId}, Logit: {maxLogit:F4})");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error processing pair: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            return (0.5f, $"Error: {ex.Message}");
        }
    }
    
    static void RunTests()
    {
        var queries = new string[]
        {
            "What is the capital of China?",
            "Explain gravity",
            "C# Backend developer",
            "C# Backend developer", 
            "Unity developer"
        };

        var documents = new string[]
        {
            "A CAT LIKE MILK",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
            "Занимался разработкой новых и оптимизацией существующих backend-сервисов для корпоративной системы, обеспечивал интеграцию со сторонними провайдерами данных.¶¶Участвовал в проектировании и разработке сервисов для обновления и миграции устаревших подсистем.¶¶Стек: .NET 6 - 8, EF, MSSQL, PostgreSQL, Swagger",
            "- Руководство группой менеджеров: распределение ресурсов, контроль качества исполняемой работы, менторинг;¶- Выполнение KPI рекламных кампаний;¶- Взаимодействие с аккаунт-менеджерами, backend и frontend отделами, дизайнерами, отделом баинга, консультация аккаунт-менеджеров;¶- Собеседование кандидатов на должность трафик-менеджера;¶- Проведение performance review;¶- Запуск и ведение рекламных кампаний (Senior Traffic Manager);¶- Оптимизация рекламных кампаний по различным верификаторам;¶- Аналитика в Яндекс.Метрика, Google Analytics;¶- Создание отчетов и посткампейн-исследований.¶",
            "Занимался разработкой новых и оптимизацией существующих backend-сервисов для корпоративной системы, обеспечивал интеграцию со сторонними провайдерами данных.¶¶Участвовал в проектировании и разработке сервисов для обновления и миграции устаревших подсистем.¶¶Стек: .NET 6 - 8, EF, MSSQL, PostgreSQL, Swagger"
        };

        Console.WriteLine("=== BATCH PROCESSING MULTIPLE PAIRS ===");
        Console.WriteLine();

        var results = new List<(string query, string document, float probability, string reasoning)>();

        for (int i = 0; i < Math.Min(queries.Length, documents.Length); i++)
        {
            Console.WriteLine($"=== Processing Pair {i + 1}/{Math.Min(queries.Length, documents.Length)} ===");
            Console.WriteLine($"Query: {queries[i]}");
            Console.WriteLine($"Document: {documents[i][..Math.Min(100, documents[i].Length)]}...");
            Console.WriteLine();
            
            var (probability, reasoning) = ProcessPair(queries[i], documents[i]);
            results.Add((queries[i], documents[i], probability, reasoning));
            
            Console.WriteLine($"Relevance probability (yes): {probability:F4}");
            Console.WriteLine($"Relevance probability (no): {(1 - probability):F4}");
            Console.WriteLine($"Reasoning: {reasoning}");
            Console.WriteLine();
            Console.WriteLine("".PadRight(80, '-'));
            Console.WriteLine();
        }

        // Summary results
        Console.WriteLine("=== SUMMARY RESULTS ===");
        Console.WriteLine();
        for (int i = 0; i < results.Count; i++)
        {
            var result = results[i];
            Console.WriteLine($"Pair {i + 1}:");
            Console.WriteLine($"  Query: {result.query}");
            Console.WriteLine($"  Document: {result.document[..Math.Min(50, result.document.Length)]}...");
            Console.WriteLine($"  Yes Probability: {result.probability:F4}");
            Console.WriteLine($"  Reasoning: {result.reasoning}");
            Console.WriteLine();
        }
    }
}