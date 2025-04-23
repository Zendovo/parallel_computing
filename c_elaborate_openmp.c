#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <omp.h>

#define MAX_LINE_SIZE 10000
#define MAX_ID_SIZE 100
#define MAX_TEXT_SIZE 5000
#define HASH_SIZE 10007
#define MIN_WORD_COUNT 50
#define MIN_REVIEW_COUNT 5
#define MAX_LINES 20000000

// Structure to store reviewer information
typedef struct ReviewerNode {
    char reviewerID[MAX_ID_SIZE];
    int elaborateReviewCount;
    struct ReviewerNode* next;
} ReviewerNode;

// Hash table with locks for parallel access
ReviewerNode* hashTable[HASH_SIZE];
omp_lock_t hashLocks[HASH_SIZE];

// Hash function for reviewer ID
unsigned int hash(const char* reviewerID) {
    unsigned int hash = 0;
    for (int i = 0; reviewerID[i] != '\0'; i++) {
        hash = hash * 31 + reviewerID[i];
    }
    return hash % HASH_SIZE;
}

// Count words in a text - thread safe
int countWords(const char* text) {
    int count = 0;
    bool inWord = false;
    
    for (int i = 0; text[i]; i++) {
        if (isspace(text[i])) {
            inWord = false;
        } else if (!inWord) {
            inWord = true;
            count++;
        }
    }
    
    return count;
}

// Add or update reviewer in hash table
void updateReviewer(const char* reviewerID, bool isElaborate) {
    if (strlen(reviewerID) == 0) return;
    
    unsigned int index = hash(reviewerID);
    
    // Lock the specific hash bucket
    omp_set_lock(&hashLocks[index]);
    
    ReviewerNode* current = hashTable[index];
    
    // Check if reviewer already exists
    while (current != NULL) {
        if (strcmp(current->reviewerID, reviewerID) == 0) {
            if (isElaborate) {
                current->elaborateReviewCount++;
            }
            omp_unset_lock(&hashLocks[index]);
            return;
        }
        current = current->next;
    }
    
    // If not found, create new entry
    ReviewerNode* newNode = (ReviewerNode*)malloc(sizeof(ReviewerNode));
    if (newNode == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        omp_unset_lock(&hashLocks[index]);
        exit(1);
    }
    
    strncpy(newNode->reviewerID, reviewerID, MAX_ID_SIZE - 1);
    newNode->reviewerID[MAX_ID_SIZE - 1] = '\0';
    newNode->elaborateReviewCount = isElaborate ? 1 : 0;
    newNode->next = hashTable[index];
    hashTable[index] = newNode;
    
    omp_unset_lock(&hashLocks[index]);
}

// Thread-safe CSV parsing function
void parseCSVLine(char* line) {
    // Make a local copy of the line for thread safety
    char line_copy[MAX_LINE_SIZE];
    strncpy(line_copy, line, MAX_LINE_SIZE - 1);
    line_copy[MAX_LINE_SIZE - 1] = '\0';
    
    char reviewerID[MAX_ID_SIZE] = "";
    char reviewText[MAX_TEXT_SIZE] = "";
    
    // Manual CSV parsing
    bool inQuotes = false;
    int field = 0;
    int idPos = 0, textPos = 0;
    
    for (int i = 0; line_copy[i] != '\0'; i++) {
        char c = line_copy[i];
        
        if (c == '"') {
            inQuotes = !inQuotes;
            continue;
        }
        
        if (c == ',' && !inQuotes) {
            field++;
            continue;
        }
        
        // Store character in appropriate field
        if (field == 0 && idPos < MAX_ID_SIZE - 1) {
            reviewerID[idPos++] = c;
        } else if (field >= 1 && textPos < MAX_TEXT_SIZE - 1) {
            // All fields after the first one are considered part of the review text
            reviewText[textPos++] = c;
        }
    }
    
    reviewerID[idPos] = '\0';
    reviewText[textPos] = '\0';
    
    
    // Count words and update reviewer
    if (strlen(reviewerID) > 0) {
        int wordCount = countWords(reviewText);
        bool isElaborate = wordCount >= MIN_WORD_COUNT;
        updateReviewer(reviewerID, isElaborate);
    }
}

// Print elaborate reviewers
void printElaborateReviewers() {
    printf("Elaborate Reviewers (with %d+ words in at least %d reviews):\n", MIN_WORD_COUNT, MIN_REVIEW_COUNT);
    int count = 0;
    
    for (int i = 0; i < HASH_SIZE; i++) {
        ReviewerNode* current = hashTable[i];
        while (current != NULL) {
            if (current->elaborateReviewCount >= MIN_REVIEW_COUNT) {
                printf("%s: %d elaborate reviews\n", current->reviewerID, current->elaborateReviewCount);
                count++;
            }
            current = current->next;
        }
    }
    
    printf("Total elaborate reviewers found: %d\n", count);
}

// Free hash table memory
void cleanupHashTable() {
    for (int i = 0; i < HASH_SIZE; i++) {
        omp_set_lock(&hashLocks[i]);
        ReviewerNode* current = hashTable[i];
        while (current != NULL) {
            ReviewerNode* temp = current;
            current = current->next;
            free(temp);
        }
        hashTable[i] = NULL;
        omp_unset_lock(&hashLocks[i]);
        omp_destroy_lock(&hashLocks[i]);
    }
}

// Read file content into memory
int readFileLines(const char* filename, char** lines, int maxLines) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return 0;
    }
    
    int lineCount = 0;
    char buffer[MAX_LINE_SIZE];
    
    // Read and discard header line
    if (fgets(buffer, sizeof(buffer), file) == NULL) {
        printf("Error reading header or file is empty\n");
        fclose(file);
        return 0;
    }
    
    // Read data lines
    while (lineCount < maxLines && fgets(buffer, sizeof(buffer), file) != NULL) {
        size_t len = strlen(buffer);
        
        // Trim trailing newline if present
        if (len > 0 && buffer[len-1] == '\n') {
            buffer[len-1] = '\0';
        }
        
        lines[lineCount] = (char*)malloc(len + 1);
        if (lines[lineCount] == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            fclose(file);
            return lineCount;
        }
        strcpy(lines[lineCount], buffer);
        lineCount++;
    }
    
    fclose(file);
    return lineCount;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <csv_file>\n", argv[0]);
        return 1;
    }
    
    double start_time, end_time;
    start_time = omp_get_wtime();
    
    // Initialize hash table and locks
    for (int i = 0; i < HASH_SIZE; i++) {
        hashTable[i] = NULL;
        omp_init_lock(&hashLocks[i]);
    }
    
    // Read file lines into memory
    char** lines = (char**)malloc(MAX_LINES * sizeof(char*));
    if (lines == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    int lineCount = readFileLines(argv[1], lines, MAX_LINES);
    if (lineCount == 0) {
        free(lines);
        return 1;
    }
    
    printf("Processing %d reviews in parallel...\n", lineCount);
    
    // Set number of threads based on available cores
    int num_threads = omp_get_max_threads();
    printf("Using %d threads\n", num_threads);
    
    // Process lines in parallel
    #pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < lineCount; i++) {
        parseCSVLine(lines[i]);
        free(lines[i]); 
    }
    
    free(lines); 
    
    printElaborateReviewers();
    
    cleanupHashTable();
    
    end_time = omp_get_wtime();
    printf("Execution time: %.4f seconds\n", end_time - start_time);
    
    return 0;
}