import java.util.ArrayList;
import java.util.Scanner;

// Storing a string an a boolean. If it's already filtered then dunnid to go through algorithm
class Word
{
    public Boolean filtered = false;
    public String w;
}

class WordFilter
{
    public static String Filter(String msg)
    {
        // This is so I can seperate each word in the message
        Scanner scanner = new Scanner(msg);

        // Adding each words into a "vector"
        ArrayList<Word> words = new ArrayList<>();
        while (scanner.hasNext())
        {
            Word word = new Word();
            word.w = scanner.next();
            words.add(word);
        }
        System.out.println(words.size());
        scanner.close();

        FilterProfanity(words);
        FilterNRIC(words);
        
        return "";
    }
    
    private static void FilterProfanity(ArrayList<Word> words)
    {

    }
    
    private static void FilterNRIC(ArrayList<Word> words)
    {
        
    }
}